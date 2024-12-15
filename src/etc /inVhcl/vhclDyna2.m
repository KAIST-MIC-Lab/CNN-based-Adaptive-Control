function grad_x = vhclDyna2(x, u, t)

%% PARAMETERS LOAD

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
    B = 15;
    C = 1.5;
    D = 1000;
    E = 0.8;
else
    B = 15;
    C = 1.5;
    D = 8000;
    E = 0.2;
end

%%
slipRate = u;
x_dot = x;


%%
Fx = D*sin(C*atan(B*slipRate-E*(B*slipRate-atan(B*slipRate))));

%% VEHICLE DYNAMICS


%%
grad_x = (Fx)/m;

end