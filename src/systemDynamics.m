function x_dot = systemDynamics(x, u, Plant, Ref)
    % x = [e_y, e_y_dot, e_psi, e_psi_dot, theta_1, theta_2]
    % u = [w_1, w_2] (Steering input rates)

    %% PHYSICAL PARAMETERS
    m    = Plant.m;    
    Iz   = Plant.Iz;   
    l1   = Plant.l1;   
    l2   = Plant.l2;   
    C1   = Plant.C1;   
    C2   = Plant.C2;  
    Fmax = Plant.Fmax; 
    vx   = Plant.vx;   
    tau  = Plant.tau;                                             % ACTUATOR TIME CONSTANT

    %% REFERENCE INPUTS
    psi_dot_des  = Ref.psi_dot_des;                               % DESIRED YAW RATE
    psi_ddot_des = Ref.psi_ddot_des;                              % DESIRED YAW ACCELERATION

    %% STATE EXTRACTION
    ey        = x(1);                                             % LATERAL ERROR
    ey_dot    = x(2);                                             % LATERAL ERROR RATE
    e_psi     = x(3);                                             % HEADING ERROR
    e_psi_dot = x(4);                                             % HEADING ERROR RATE
    theta_1   = x(5);                                             % FRONT WHEEL STEER ANGLE
    theta_2   = x(6);                                             % REAR WHEEL STEER ANGLE

    %% SLIP ANGLE CALCULATION 
    alpha_1 = theta_1 - (ey_dot - vx*e_psi + l1*(e_psi_dot + psi_dot_des)) / vx;
    alpha_2 = theta_2 - (ey_dot - vx*e_psi - l2*(e_psi_dot + psi_dot_des)) / vx;

    %% NON-LINEAR TIRE FORCE MODEL 
    F_1 = Fmax * tanh((C1 / Fmax) * alpha_1);                     % FRONT TIRE FORCE
    F_2 = Fmax * tanh((C2 / Fmax) * alpha_2);                     % REAR TIRE FORCE

    %% GOVERNING ERROR DYNAMICS 
    ey_ddot    = (F_1 + F_2)/m + vx*e_psi_dot - vx*psi_dot_des;   % LATERAL ERROR ACCELERATION
    e_psi_ddot = (l1*F_1 - l2*F_2)/Iz - psi_ddot_des;             % YAW ERROR ACCELERATION

    %% ACTUATOR'S LAG DYNAMICS
    theta_1_dot = (u(1) - theta_1) / tau;                         % FRONT ACTUATOR DYNAMICS
    theta_2_dot = (u(2) - theta_2) / tau;                         % REAR ACTUATOR DYNAMICS

    %% DERIVATIVE VECTOR CONSTRUCTION
    x_dot = [ey_dot;      ...                                     % X1_DOT
             ey_ddot;     ...                                     % X2_DOT
             e_psi_dot;   ...                                     % X3_DOT
             e_psi_ddot;  ...                                     % X4_DOT
             theta_1_dot; ...                                     % X5_DOT
             theta_2_dot];                                        % X6_DOT
end