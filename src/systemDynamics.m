function x_dot = systemDynamics(x, u, Plant)
% x = [Y (Global), vy (body), Psi (Global), r (body), beta, theta_f, theta_r]
% u = [w_f, w_r] (Steering input rates)

%% Physical Parameters from Plant Structure
m    = Plant.m;    
Iz   = Plant.Iz;   
lf   = Plant.lf;   
lr   = Plant.lr;   
Cf   = Plant.Cf;   
Cr   = Plant.Cr;  
Fmax = Plant.Fmax; 
vx   = Plant.vx;  
delay_time= Plant.delay_time;  

%% State Extraction
vy      = x(2); 
Psi     = x(3); 
r       = x(4); 
beta    = x(5);
theta_f = x(6); 
theta_r = x(7); 

%% Non-Linear Tire Force Model 
alpha_f = beta + lf*r/vx - theta_f; 
alpha_r = beta - lr*r/vx - theta_r;

% F = Fmax * tanh( (C/Fmax) * alpha ) 
F_f = -Fmax * tanh((Cf/Fmax) * alpha_f); 
F_r = -Fmax * tanh((Cr/Fmax) * alpha_r);

%% Governing Rate Equations
Y_dot = vy*cos(Psi) + vx*sin(Psi); 
Psi_dot = r; 
vy_dot = (F_f + F_r)/m - vx*r; 
r_dot = (lf*F_f - lr*F_r)/Iz; 
beta_dot = (F_f + F_r)/(m*vx) - r; 
theta_f_dot = (u(1)-theta_f)/delay_time;
theta_r_dot = (u(2)-theta_r)/delay_time; 

%% Derivative Vector Construction
x_dot = [Y_dot; vy_dot; Psi_dot; r_dot;...
        beta_dot; theta_f_dot; theta_r_dot];

end