function tire_forces = tire_model(theta,state, vehicle_params)
    % theta: effective steering angle
    % state: [y, psi, vy, r, vx]

    % Extract parameters
    C1 = vehicle_params.C1;
    C2 = vehicle_params.C2;
    Fmax = vehicle_params.Fmax;
    l1= vehicle_params.l1;
    l2= vehicle_params.l2;
    
    % Extract state variables
    vx=state.vx;
    vy=state.vy;
    r=state.r;  
    theta_1 = theta.theta_1;
    theta_2 = theta.theta_2;

    % Calculate slip angles
    alpha_1= theta_1 - atan((vy + l1*r) / vx);
    alpha_2= theta_2 - atan((vy - l2*r) / vx);

    % Tanh tire model
    F_1= Fmax * tanh(C1 * alpha_1);
    F_2 = Fmax * tanh(C2 * alpha_2);
    tire_forces.F_1 = F_1;
    tire_forces.F_2 = F_2;
end