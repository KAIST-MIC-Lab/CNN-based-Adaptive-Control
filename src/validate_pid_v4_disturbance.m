%% validate_pid_v4_disturbance.m
clear;
clc;
close all;

%% PARAMETERS
sim.dt = 0.01;
sim.T_end = 27.2;
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

ctrl.front.Kp_y = -0.6;
ctrl.front.Kp_psi = 0.85;

% Rear steering supports the front axle with opposite sign.
ctrl.rear.support_ratio = 0.35;

ctrl.steer_limit = 0.17;

% Disturbance model: external lateral force and yaw moment.
disturb.enable = true;
disturb.t_start = 6.0;
disturb.t_end = 15.0;
disturb.Fy_bias = 450;
disturb.Fy_amp = 250;
disturb.Fy_freq = 0.6;
disturb.Mz_bias = 180;
disturb.Mz_amp = 120;
disturb.Mz_freq = 0.9;

%% INITIAL CONDITIONS
% state = [X_v; Y_v; psi; v_y; r; theta_f; theta_r]
state = [0; -1; 0; 0; 0; 0; 0];

%% LOGGING VARIABLES
N = numel(sim.t);
state_hist = zeros(7, N);
input_hist = zeros(2, N);
error_hist = zeros(2, N);
proj_hist = zeros(2, N);
psi_ref_hist = zeros(1, N);
dist_hist = zeros(2, N);

%% SIMULATION LOOP
for k = 1:N
    Xv = state(1);
    Yv = state(2);
    psi = state(3);
    t_now = sim.t(k);

    ref = project_region_based(Xv, Yv, track);
    ey = ref.ey;
    epsi = wrap_to_pi(ref.psi_des - psi);

    omega_1 = ...
        ctrl.front.Kp_y * ey + ...
        ctrl.front.Kp_psi * epsi;

    omega_2 = -ctrl.rear.support_ratio * omega_1;

    omega_1 = saturate(omega_1, ctrl.steer_limit);
    omega_2 = saturate(omega_2, ctrl.steer_limit);

    [state_dot, disturbance_now] = plant_dynamics(state, [omega_1; omega_2], vehicle, disturb, t_now);
    state = state + sim.dt * state_dot;

    state_hist(:, k) = state;
    input_hist(:, k) = [omega_1; omega_2];
    error_hist(:, k) = [ey; epsi];
    proj_hist(:, k) = [ref.X_star; ref.Y_star];
    psi_ref_hist(k) = ref.psi_des;
    dist_hist(:, k) = disturbance_now;
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
title("4WS Tracking with External Disturbance");
animate_tracking(state_hist, proj_hist, track_plot, vehicle, sim);

figure("Color", "w");
subplot(4, 1, 1);
plot(sim.t, error_hist(1, :), "LineWidth", 1.4);
grid on;
ylabel("e_y [m]");
title("Tracking Errors, Steering Inputs, and Disturbance");

subplot(4, 1, 2);
plot(sim.t, error_hist(2, :), "LineWidth", 1.4);
grid on;
ylabel("e_\psi [rad]");

subplot(4, 1, 3);
plot(sim.t, input_hist(1, :), "LineWidth", 1.4);
hold on;
plot(sim.t, input_hist(2, :), "LineWidth", 1.4);
grid on;
ylabel("\omega [rad]");
legend("\omega_1", "\omega_2", "Location", "best");

subplot(4, 1, 4);
yyaxis left;
plot(sim.t, dist_hist(1, :), "LineWidth", 1.4);
ylabel("F_{y,d} [N]");
yyaxis right;
plot(sim.t, dist_hist(2, :), "LineWidth", 1.4);
grid on;
xlabel("Time [s]");
ylabel("M_{z,d} [N m]");
legend("Lateral disturbance", "Yaw disturbance", "Location", "best");

fprintf("validate_pid_v4_disturbance.m finished.\n");
fprintf("RMSE lateral error : %.4f m\n", sqrt(mean(error_hist(1, :).^2)));
fprintf("RMSE heading error : %.4f rad\n", sqrt(mean(error_hist(2, :).^2)));


%% FUNCTION DEFINITIONS

function [state_dot, disturbance_vec] = plant_dynamics(state, omega_cmd, vehicle, disturb, t_now)
    C1 = vehicle.C1;
    C2 = vehicle.C2;
    l1 = vehicle.l1;
    l2 = vehicle.l2;
    vx = vehicle.vx;
    m  = vehicle.m;
    Iz = vehicle.Iz;
    Fmax = vehicle.Fmax;
    tau = vehicle.tau;

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

    [Fy_dist, Mz_dist] = disturbance_profile(t_now, disturb);

    vy_dot = (Fy_f + Fy_r + Fy_dist) / m - vx * r;
    r_dot = (l1 * Fy_f - l2 * Fy_r + Mz_dist) / Iz;

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

    disturbance_vec = [Fy_dist; Mz_dist];
end

function [Fy_dist, Mz_dist] = disturbance_profile(t_now, disturb)
    Fy_dist = 0;
    Mz_dist = 0;

    if ~disturb.enable
        return;
    end

    if t_now < disturb.t_start || t_now > disturb.t_end
        return;
    end

    t_shift = t_now - disturb.t_start;

    Fy_dist = ...
        disturb.Fy_bias + ...
        disturb.Fy_amp * sin(2 * pi * disturb.Fy_freq * t_shift);

    Mz_dist = ...
        disturb.Mz_bias + ...
        disturb.Mz_amp * sin(2 * pi * disturb.Mz_freq * t_shift + pi / 6);
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
    ref.ey = Yv;
end

function ref = upper_straight_projection(Xv, Yv, R)
    ref.X_star = Xv;
    ref.Y_star = 2 * R;
    ref.psi_des = pi;
    ref.ey = 2 * R - Yv;
end

function ref = right_arc_projection(Xv, Yv, L, R)
    dx = Xv - L;
    dy = Yv - R;
    theta = atan2(dy, dx);

    ref.X_star = L + R * cos(theta);
    ref.Y_star = R + R * sin(theta);
    ref.psi_des = wrap_to_pi(theta + pi / 2);
    ref.ey = R - hypot(dx, dy);
end

function ref = left_arc_projection(Xv, Yv, R)
    dx = Xv;
    dy = Yv - R;
    theta = atan2(dy, dx);

    ref.X_star = R * cos(theta);
    ref.Y_star = R + R * sin(theta);
    ref.psi_des = wrap_to_pi(theta + pi / 2);
    ref.ey = R - hypot(dx, dy);
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

function animate_tracking(state_hist, proj_hist, track_plot, vehicle, sim)
    body_length = vehicle.l1 + vehicle.l2;
    body_width = 1.8;
    axle_width = 1.6;
    heading_length = 1.8;
    step = max(1, round(0.03 / sim.dt));

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
        theta_f = state_hist(6, k);
        theta_r = state_hist(7, k);

        set(traj_handle, "XData", state_hist(1, 1:k), "YData", state_hist(2, 1:k));
        set(proj_handle, "XData", proj_hist(1, k), "YData", proj_hist(2, k));
        set(vehicle_center_handle, "XData", Xv, "YData", Yv);

        [body_x, body_y] = vehicle_body_polygon(Xv, Yv, psi, body_length, body_width);
        set(body_handle, "XData", body_x, "YData", body_y);

        heading_x = [Xv, Xv + heading_length * cos(psi)];
        heading_y = [Yv, Yv + heading_length * sin(psi)];
        set(heading_handle, "XData", heading_x, "YData", heading_y);

        [front_x, front_y] = axle_segment(Xv, Yv, psi, vehicle.l1, axle_width, theta_f);
        [rear_x, rear_y] = axle_segment(Xv, Yv, psi, -vehicle.l2, axle_width, theta_r);
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
