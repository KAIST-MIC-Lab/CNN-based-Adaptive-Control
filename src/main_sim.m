clear;
close all;
clc;
addpath(genpath('./utils/'));
addpath(genpath('./controllers/'));
rng(10); % For reproducibility


%% Simulation Parameters
sim_params.dt = 0.001;                                  %  1 kHz 
sim_params.T = 27.2;                                      % Total simulation time
sim_params.time = 0:sim_params.dt:sim_params.T;         % Time vector
sim_params.N = length(sim_params.time);                 % Simulation steps

%% Vehicle Parameters
vehicle_params= vehicle_params();                       % Load vehicle

%% Controller Parameters
ctrl_param.dt  = 0.01;                                  % 100Hz control
ctrl_param.Q = diag([20, 20]);                          % State tracking cost
ctrl_param.R = diag([5, 5]);                            % Control effort cost
ctrl_param.eta = 0.001;                                 % Learning rate
lstm_param = lstm_params();                             % Load LSTM parameters
ctrl_param.lstm_param = lstm_param;                     % Embed LSTM params in
ctrl_param.max_ey = 1;                                % Max expected lateral error (meters)
ctrl_param.max_epsi = 1;                              % Max expected heading error (radians)

%% Initializations
% Initialize lstm
lstm= init_lstm(lstm_param);

% Initial state [X_v; Y_v; psi; v_y; r]
state = [0; -1; 0; 0; 0];
% longitudinal speed

% Initialize track
track.L = 20;
track.R = 15;

%% LOGGING VARIABLES
N = numel(sim_params.time);
state_hist = zeros(5, N);
input_hist = zeros(2, N);
error_hist = zeros(2, N);
proj_hist = zeros(2, N);
psi_ref_hist = zeros(1, N);
norm_hist = zeros(6, N);

%% SIMULATION LOOP
for t_idx = 1:N
    Xv = state(1);
    Yv = state(2);
    psi = state(3);

    % Get reference trajectory and errors
    ref = get_ref(Xv, Yv, track);
    ey = ref.ey;
    epsi = wrap_to_pi(ref.psi_des - psi);
    e=[ey; epsi];

    
    if mod(t_idx-1, 10) == 0 || t_idx == 1
        % lstm forward pass
        % lSTM input construction
        x = [e(1)/ctrl_param.max_ey; 
                  e(2)/ctrl_param.max_epsi];
        [Phi, lstm] = lstm_forward(ctrl_param, lstm, x); 

        % train lstm
        lstm = lstm_train(ctrl_param, lstm, e, Phi);
    end
    u= Phi;                                                 % end to end

    state_dot = state_dynamics(state, u, vehicle_params);
    state = state + sim_params.dt * state_dot;

    state_hist(:, t_idx) = state;
    input_hist(:, t_idx) = u;
    error_hist(:, t_idx) = [ey; epsi];
    proj_hist(:, t_idx) = [ref.X_star; ref.Y_star];
    psi_ref_hist(t_idx) = ref.psi_des;

    % Calculate Frobenius norms for each weight matrix
    norm_hist(1, t_idx) = norm(lstm.Wf, 'fro');
    norm_hist(2, t_idx) = norm(lstm.Wi, 'fro');
    norm_hist(3, t_idx) = norm(lstm.Wo, 'fro');
    norm_hist(4, t_idx) = norm(lstm.Wc, 'fro');
    norm_hist(5, t_idx) = norm(lstm.Wh, 'fro');
    norm_hist(6, t_idx) = norm(lstm.Wff, 'fro');
end
%% REFERENCE TRACK
track_plot = build_stadium_track(track, 200);

%% ANIMATION AND PLOTS
figure("Color", "w");
plot(track_plot(1, :), track_plot(2, :), "k--", "LineWidth", 1.5);
hold on;
axis equal;
grid on;
xlabel("X [m]");
ylabel("Y [m]");
title("Animated 4WS Tracking with Current Vehicle Orientation");
animate_tracking(state_hist, proj_hist, track_plot, vehicle_params, sim_params);

figure("Color", "w");
subplot(3, 1, 1);
plot(sim_params.time, error_hist(1, :), "LineWidth", 1.4);
grid on;
ylabel("e_y [m]");
title("Tracking Errors and 4WS Steering Commands");

subplot(3, 1, 2);
plot(sim_params.time, error_hist(2, :), "LineWidth", 1.4);
grid on;
ylabel("e_\psi [rad]");

subplot(3, 1, 3);
plot(sim_params.time, input_hist(1, :), "LineWidth", 1.4);
hold on;
plot(sim_params.time, input_hist(2, :), "LineWidth", 1.4);
grid on;
xlabel("Time [s]");
ylabel("\delta [rad]");
legend("\delta_f", "\delta_r", "Location", "best");

fprintf("LSTM-ADAPTIVE PATH TRACKING SIMULATION RESULTS:\n");
fprintf("RMSE lateral error : %.4f m\n", sqrt(mean(error_hist(1, :).^2)));
fprintf("RMSE heading error : %.4f rad\n", sqrt(mean(error_hist(2, :).^2)));

figure("Color", "w", "Name", "LSTM Weight Convergence");
plot(sim_params.time, norm_hist', "LineWidth", 1.5);
grid on;
xlabel("Time [s]");
ylabel("Frobenius Norm ||W||_F");
legend("W_f (Forget)", "W_i (Input)", "W_o (Output)", ...
       "W_c (Cell)", "W_h (Mapping)", "W_{ff} (Feedforward)", ...
       "Location", "best", "NumColumns", 2);
title("Evolution of LSTM Adaptive Weights");

%% VEHICLE DYNAMICS
function state_dot = state_dynamics(state, u, vehicle_params)  
    % Assumptions: No actuator lag, no longitudinal dynamics, constant longitudinal velocity (vx).
    
    % Unpack vehicle parameters
    C1 = vehicle_params.C1;
    C2 = vehicle_params.C2;
    l1 = vehicle_params.l1;
    l2 = vehicle_params.l2;
    m  = vehicle_params.m;
    Iz = vehicle_params.Iz;
    Fmax= vehicle_params.Fmax;
    vx = vehicle_params.vx;

    % Unpack state 
    psi = state(3);
    vy = state(4);
    r = state(5);
    alpha_1 = u(1) - (vy + l1 * r) / vx;
    alpha_2 = u(2) - (vy - l2 * r) / vx;

    Fy_f = Fmax * tanh((C1 / Fmax) * alpha_1);
    Fy_r = Fmax * tanh((C2 / Fmax) * alpha_2);

    vy_dot = (Fy_f + Fy_r) / m - vx * r;
    r_dot = (l1 * Fy_f - l2 * Fy_r) / Iz;

    Xv_dot = vx * cos(psi) - vy * sin(psi);
    Yv_dot = vx * sin(psi) + vy * cos(psi);
    psi_dot = r;


    state_dot = [
        Xv_dot;
        Yv_dot;
        psi_dot;
        vy_dot;
        r_dot;
    ];
end

function angle = wrap_to_pi(angle)
    angle = mod(angle + pi, 2 * pi) - pi;
end

%% VISUALIZATION
function path_xy = build_stadium_track(track, n)
    x_bottom = linspace(0, track.L, n);
    y_bottom = zeros(1, n);

    th_right = linspace(-pi / 2, pi / 2, n);
    x_right = track.L + track.R * cos(th_right);
    y_right = track.R + track.R * sin(th_right);

    x_top = linspace(track.L, 0, n);
    y_top = 2 * track.R * ones(1, n);

    th_left = linspace(pi / 2, 3 * pi / 2, n);
    x_left = track.R * cos(th_left);
    y_left = track.R + track.R * sin(th_left);

    path_xy = [
        [x_bottom, x_right, x_top, x_left];
        [y_bottom, y_right, y_top, y_left]
    ];
end

% ANIMATION FUNCTIONS
function animate_tracking(state_hist, proj_hist, track_plot, vehicle_params, sim_params)
    body_length = vehicle_params.l1 + vehicle_params.l2;
    body_width = 1.8;
    axle_width = 1.6;
    heading_length = 1.8;
    step = max(1, round(0.03 / sim_params.dt));

    traj_handle = plot(nan, nan, "b", "LineWidth", 1.8);
    proj_handle = plot(nan, nan, "r.", "MarkerSize", 10);
    body_handle = patch(nan, nan, [0.3 0.6 0.9], ...
        "FaceAlpha", 0.18, "EdgeColor", [0 0.2 0.7], "LineWidth", 1.4);
    heading_handle = plot(nan, nan, "b-", "LineWidth", 2.2);
    front_axle_handle = plot(nan, nan, "Color", [0 0.45 0.74], "LineWidth", 2.0);
    rear_axle_handle = plot(nan, nan, "Color", [0.85 0.33 0.10], "LineWidth", 2.0);
    vehicle_center_handle = plot(nan, nan, "bo", ...
        "MarkerFaceColor", "b", "MarkerSize", 5);

    legend([traj_handle, proj_handle, heading_handle], ...
        {"Vehicle trajectory", "Projected point", "Current orientation"}, ...
        "Location", "best");

    x_margin = 5;
    y_margin = 5;
    xlim([min(track_plot(1, :)) - x_margin, max(track_plot(1, :)) + x_margin]);
    ylim([min(track_plot(2, :)) - y_margin, max(track_plot(2, :)) + y_margin]);

    for k = 1:step:size(state_hist, 2)
        Xv = state_hist(1, k);
        Yv = state_hist(2, k);
        psi = state_hist(3, k);
        theta_f = state_hist(4, k);
        theta_r = state_hist(5, k);

        set(traj_handle, "XData", state_hist(1, 1:k), "YData", state_hist(2, 1:k));
        set(proj_handle, "XData", proj_hist(1, k), "YData", proj_hist(2, k));
        set(vehicle_center_handle, "XData", Xv, "YData", Yv);

        [body_x, body_y] = vehicle_body_polygon(Xv, Yv, psi, body_length, body_width);
        set(body_handle, "XData", body_x, "YData", body_y);

        heading_x = [Xv, Xv + heading_length * cos(psi)];
        heading_y = [Yv, Yv + heading_length * sin(psi)];
        set(heading_handle, "XData", heading_x, "YData", heading_y);

        [front_x, front_y] = axle_segment(Xv, Yv, psi, vehicle_params.l1, axle_width, theta_f);
        [rear_x, rear_y] = axle_segment(Xv, Yv, psi, -vehicle_params.l2, axle_width, theta_r);
        set(front_axle_handle, "XData", front_x, "YData", front_y);
        set(rear_axle_handle, "XData", rear_x, "YData", rear_y);

        drawnow;
        pause(0.01);
    end
end

function [x_poly, y_poly] = vehicle_body_polygon(Xv, Yv, psi, length_body, width_body)
    corners = [
        length_body / 2, width_body / 2;
        length_body / 2, -width_body / 2;
        -length_body / 2, -width_body / 2;
        -length_body / 2, width_body / 2
    ]';

    R = [cos(psi), -sin(psi); sin(psi), cos(psi)];
    world_corners = R * corners + [Xv; Yv];

    x_poly = [world_corners(1, :), world_corners(1, 1)];
    y_poly = [world_corners(2, :), world_corners(2, 1)];
end

function [x_axle, y_axle] = axle_segment(Xv, Yv, psi, axle_offset, axle_width, steer_angle)
    axle_center = [
        Xv + axle_offset * cos(psi);
        Yv + axle_offset * sin(psi)
    ];

    axle_heading = psi + steer_angle;
    half_width = axle_width / 2;

    x_axle = axle_center(1) + half_width * [-cos(axle_heading + pi / 2), cos(axle_heading + pi / 2)];
    y_axle = axle_center(2) + half_width * [-sin(axle_heading + pi / 2), sin(axle_heading + pi / 2)];
end