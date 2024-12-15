function grad_x = vhclDyna(x, u, t)

%% PARAMETERS LOAD
Vx = 15;

% Cf = 2.1827e3 ;
% Cr = 6.2533e4 ;

% reff = 0.033;
% reff = 0.0454;

g = 9.81;
h = 0.573;
Iz = 2815;
Iy = 2700;
Ix = 530;
lf = 1.3;
lr = 1.4;
m = 1426;
w = 1.5/2;
Cd = 0.38;
Ad = 2.7;

if t < 3
    B = 10;
    C = 1.3;
    D = 2500;
    E = 0.1;
else
    B = 8;
    C = 1.5;
    D = 1500;
    E = 0.2;
end

%%
delta = u;
x_dot = Vx;
y_dot = x(1);
psi_dot = x(2);

%%
alpha_f = delta - (y_dot + lf * psi_dot) / x_dot;
alpha_r = - (y_dot - lr * psi_dot) / x_dot;

Fyf = D*sin(C*atan(B*alpha_f-E*(B*alpha_f-atan(B*alpha_f))));
Fyr = D*sin(C*atan(B*alpha_r-E*(B*alpha_r-atan(B*alpha_r))));

%% VEHICLE DYNAMICS


%%
grad_x = [
    (Fyf+Fyr)/m
    (Fyf*lf-Fyr*lr)/Iz
    ];

end