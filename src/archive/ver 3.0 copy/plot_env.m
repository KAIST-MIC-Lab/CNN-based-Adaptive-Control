function X_grey = plot_env(x, env, NN)

if ~NN.paramCtrl.CVL2Don
    X_grey = [];
    return
end

size_d = NN.paramCtrl.size_CVL_input;

%%
angle = x(1);
L = env.l;
w = env.w;
plot_ratio = 1.25; % x, y limitation

%%
R = @(th) [cos(th) -sin(th); sin(th) cos(th)];
eq_pts = [
    -w      w       w      -w
    -w      -w      L+w    L+w
];

%%

fig = figure(9);clf
pts = R(angle) * eq_pts;


fill(pts(1,:), pts(2,:), 'r')
axis off
xlim([-L-w, L+w] * plot_ratio);
ylim([-L-w, L+w] * plot_ratio);
set(gcf,'color','w');
% figure_width_pixels = 128; % Set the desired width in pixels
% figure_height_pixels = 128; % Set the desired height in pixels    
% set(fig, 'Position', [0, 0, figure_width_pixels, figure_height_pixels])

F = getframe(gcf);
[X, Map] = frame2im(F);
X = imresize(X, size_d);

X_grey = rgb2gray(X);
end



