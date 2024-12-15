function dyn = vhclModel()

%% PARAMETERS LOAD
Vx = 15;

% Cf = 2.1827e3 ;
% Cr = 6.2533e4 ;


Cf = 2000;
Cr =  2000;

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

B = 10;
C = 1.3;
D = 2000;
E = 0.1;

%% VEHICLE DYNAMICS
a11 = -(Cf+Cr)/m/Vx;
a12 = -(Cf*lf-Cr*lr)/m/Vx-Vx;
a21 = -(Cf*lf-Cr*lr)/Iz/Vx;
a22 = -(Cf*lf^2+Cr*lr^2)/Iz/Vx;

b1 = Cf/m;
b2 = Cf*lf/Iz;

A = [a11 a12;a21 a22];
B = [b1;b2];

invB = pinv(B);

%%
dyn.A = A;
dyn.B = B;
dyn.invB = invB;

end