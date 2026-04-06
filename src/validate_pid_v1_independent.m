%% validate_pid.m
clear;
clc;
close all;

%% Parameters
sim.dt = 0.01;
sim.T_end = 15;
sim.t = 0:sim.dt:sim.T_end;

vehicle.m = 1500;
vehicle.Iz = 2500;
vehicle.l1 = 1.10;
vehicle.l2 = 1.60;
vehicle.C1 = 90000;
vehicle.C2 = 90000;
vehicle.Fmax = 8000;
vehicle.tau = 0.10;
vehicle.vx = 5.0;

track.L = 20;
track.R = 15;

ctrl.front.Kp_y = -0.24;
ctrl.front.Ki_y = 0;
ctrl.front.Kp_psi = 0;
ctrl.front.Ki_psi = 0;

ctrl.rear.Kp_y = 0;
ctrl.rear.Ki_y = 0;
ctrl.rear.Kp_psi = 0;
ctrl.rear.Ki_psi = 0;

ctrl.steer_limit = 0.17;
ctrl.int_ey_limit = 4.0;
ctrl.int_epsi_limit = 0.8;

%% Initial state
% state = [X_v; Y_v; psi; v_y; r; theta_f; theta_r]
state = [0; 0; 0; 0; 0; 0; 0];
int_ey = 0;
int_epsi = 0;

%% Logging
N = numel(sim.t);
state_hist = zeros(7, N);
input_hist = zeros(2, N);
error_hist = zeros(2, N);
proj_hist = zeros(2, N);
psi_ref_hist = zeros(1, N);

%% Simulation loop
for k = 1:N
    Xv = state(1);
    Yv = state(2);
    psi = state(3);

    ref = project_region_based(Xv, Yv, track);
    ey = ref.ey;
    epsi = wrap_to_pi(ref.psi_des - psi);

    if abs(ey) < ctrl.int_ey_limit
        int_ey = int_ey + ey * sim.dt;
    end

    if abs(epsi) < ctrl.int_epsi_limit
        int_epsi = int_epsi + epsi * sim.dt;
    end

    omega_1 = ...
        ctrl.front.Kp_y * ey + ...
        ctrl.front.Ki_y * int_ey + ...
        ctrl.front.Kp_psi * epsi + ...
        ctrl.front.Ki_psi * int_epsi;

    omega_2 = ...
        ctrl.rear.Kp_y * ey + ...
        ctrl.rear.Ki_y * int_ey + ...
        ctrl.rear.Kp_psi * epsi + ...
        ctrl.rear.Ki_psi * int_epsi;

    omega_1 = saturate(omega_1, ctrl.steer_limit);
    omega_2 = saturate(omega_2, ctrl.steer_limit);

    state_dot = plant_dynamics(state, [omega_1; omega_2], vehicle);
    state = state + sim.dt * state_dot;

    state_hist(:, k) = state;
    input_hist(:, k) = [omega_1; omega_2];
    error_hist(:, k) = [ey; epsi];
    proj_hist(:, k) = [ref.X_star; ref.Y_star];
    psi_ref_hist(k) = ref.psi_des;
end

%% Reference track
track_plot = build_stadium_track(track, 200);

%% Plots
figure("Color", "w");
plot(track_plot(1, :), track_plot(2, :), "k--", "LineWidth", 1.5);
hold on;
plot(state_hist(1, :), state_hist(2, :), "b", "LineWidth", 1.8);
plot(proj_hist(1, 1:10:end), proj_hist(2, 1:10:end), "r.", "MarkerSize", 8);
axis equal;
grid on;
xlabel("X [m]");
ylabel("Y [m]");
title("Independent Front/Rear PI with Region-Based Projection");
legend("Reference path", "Vehicle trajectory", "Projected points", ...
    "Location", "best");

figure("Color", "w");
subplot(3, 1, 1);
plot(sim.t, error_hist(1, :), "LineWidth", 1.4);
grid on;
ylabel("e_y [m]");
title("Tracking Errors and Independent Steering Commands");

subplot(3, 1, 2);
plot(sim.t, error_hist(2, :), "LineWidth", 1.4);
grid on;
ylabel("e_\psi [rad]");

subplot(3, 1, 3);
plot(sim.t, input_hist(1, :), "LineWidth", 1.4);
hold on;
plot(sim.t, input_hist(2, :), "LineWidth", 1.4);
grid on;
xlabel("Time [s]");
ylabel("\omega [rad]");
legend("\omega_1", "\omega_2", "Location", "best");

fprintf("validate_pid.m finished.\n");
fprintf("RMSE lateral error : %.4f m\n", sqrt(mean(error_hist(1, :).^2)));
fprintf("RMSE heading error : %.4f rad\n", sqrt(mean(error_hist(2, :).^2)));

function state_dot = plant_dynamics(state, omega_cmd, vehicle)
    C1 = vehicle.C1;
    C2 = vehicle.C2;
    l1 = vehicle.l1;
    l2 = vehicle.l2;
    vx = vehicle.vx;
    m  = vehicle.m;
    Iz = vehicle.Iz;
    Fmax= vehicle.Fmax;
    tau=  vehicle.tau; 
    
    psi = state(3);
    vy = state(4);
    r = state(5);
    theta_1 = state(6);
    theta_2 = state(7);

    omega_1 = omega_cmd(1);
    omega_2 = omega_cmd(2);

    alpha_1 = theta_1 - (vy + l1 * r) / vx;
    alpha_2 = theta_2 - (vy - l2 * r) / vx;

    Fy_f = Fmax * tanh((C1 / Fmax) * alpha_1);
    Fy_r = Fmax * tanh((C2 / Fmax) * alpha_2);

    vy_dot = (Fy_f + Fy_r) / m - vx * r;
    r_dot = (l1 * Fy_f - l2 * Fy_r) / Iz;

    Xv_dot = vx * cos(psi) - vy * sin(psi);
    Yv_dot = vx * sin(psi) + vy * cos(psi);
    psi_dot = r;

    theta_1_dot = (omega_1 - theta_1) / tau;
    theta_2_dot = (omega_2 - theta_2) / tau;

    state_dot = [
        Xv_dot;
        Yv_dot;
        psi_dot;
        vy_dot;
        r_dot;
        theta_1_dot;
        theta_2_dot
    ];
end

function ref = project_region_based(Xv, Yv, track)
    R = track.R;
    L = track.L;

    if Yv < R
        if Xv >= 0 && Xv <= L
            ref = lower_straight_projection(Xv, Yv);
        elseif Xv > L
            ref = right_arc_projection(Xv, Yv, L, R);
        else
            ref = left_arc_projection(Xv, Yv, R);
        end
    else
        if Xv >= 0 && Xv <= L
            ref = upper_straight_projection(Xv, Yv, R);
        elseif Xv > L
            ref = right_arc_projection(Xv, Yv, L, R);
        else
            ref = left_arc_projection(Xv, Yv, R);
        end
    end
end

function ref = lower_straight_projection(Xv, Yv)
    ref.X_star = Xv;
    ref.Y_star = 0;
    ref.psi_des = 0;

    % lateral error
    ref.ey = Yv;
end

function ref = upper_straight_projection(Xv, Yv, R)
    ref.X_star = Xv;
    ref.Y_star = 2 * R;
    ref.psi_des = pi;

    %lateral error
    ref.ey = 2 * R - Yv;
end

function ref = right_arc_projection(Xv, Yv, L, R)
    dx = Xv - L;
    dy = Yv - R;
    theta = atan2(dy, dx);

    ref.X_star = L + R * cos(theta);
    ref.Y_star = R + R * sin(theta);
    ref.psi_des = theta + pi / 2

    % lateral error
    ref.ey = R - hypot(dx, dy)
end

function ref = left_arc_projection(Xv, Yv, R)
    dx = Xv;
    dy = Yv - R;
    theta = atan2(dy, dx)
    disp(theta)

    ref.X_star = R * cos(theta);
    ref.Y_star = R + R * sin(theta);

    ref.psi_des = theta +  pi / 2;    

    % lateral error
    ref.ey = R- hypot(dx, dy);
end

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

function y = saturate(u, limit)
    y = max(min(u, limit), -limit);
end

function angle = wrap_to_pi(angle)
    angle = atan2(sin(angle), cos(angle));
end
