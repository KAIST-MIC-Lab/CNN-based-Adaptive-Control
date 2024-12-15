clear

%%
T = 10;
dt = 0.1;
t = 0:dt:T;

%%
angle = 0;
L = 5;
w = 1;
plot_ratio = 1.25;

%%
R = @(th) [cos(th) -sin(th); sin(th) cos(th)];
eq_pts = [
    -w      w       w      -w
    -w      -w      L+w    L+w
];

%%
figure(1); clf;
for t_idx = 2:length(t)
    angle = angle + 1 * dt;

    pts = R(angle) * eq_pts;


    fill(pts(1,:), pts(2,:), 'r')
    axis off
    xlim([-L-w, L+w] * plot_ratio);
    ylim([-L-w, L+w] * plot_ratio);
    set(gcf,'color','w');
    figure_width_pixels = 128; % Set the desired width in pixels
    figure_height_pixels = 128; % Set the desired height in pixels    
    set(figure(1), 'Position', [-1919, 1667, figure_width_pixels, figure_height_pixels])

    F = getframe(gcf);
    [X, Map] = frame2im(F);

    X_grey = rgb2gray(X);
end



