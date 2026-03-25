function Plant = paramPlant_load()
    % Vehicle Dynamics Parameters
    Plant.m = 1650;           % Vehicle Mass
    Plant.Iz = 315.3;         % Yaw Moment of Inertia (kg*m^2) 
    Plant.l1 = 1.11;          % Front wheelbase (m) 
    Plant.l2 = 1.59;          % Rear wheelbase (m) 
    Plant.C1 = 133000;        % Front Cornering Stiffness (N/rad)
    Plant.C2 = 98800;         % Rear Cornering Stiffness (N/rad) 
    Plant.Fmax = 8000;        % Maximum Tire Force (N) 
    Plant.vx = 5;           % Longitudinal Velocity (m/s) 
    
    % Steering Delay Configuration
    Plant.tau = 0.1;   % 100ms steering latency

end