function [] = resultReportPlot(result, paramSim)

%% SAVE?
saveResult  = paramSim.saveResult;
saveFigFile = paramSim.saveFigFile;
exp_name    = paramSim.exp_name;

%% PLOT SETTING
figure_name = [
    "state_and_ref"
    "error"
    "controls"
    "weight_norm"
    "global_trajectory"                                           % ADDED GLOBAL TRAJECTORY NAME
];

font_size       = 24;
axes_font_size  = 18;
font_name       = "Times New Roman";
line_width      = 1.5;

%% PREPARE
t          = result.t;
t_idx      = result.t_idx;
NN         = result.NN;

Y_hist     = result.Y_hist;  
YD_hist    = result.YD_hist;
U_hist     = result.U_hist;
XY_pos     = result.XY_pos;                                       % ACTUAL GLOBAL X-Y [2 x N]
XY_pos_des = result.XY_pos_des;                                   % DESIRED GLOBAL X-Y [2 x N]

%% ==============================================================
%% SIMULATION RESULT REPORT
%% ==============================================================
fprintf("===========================================\n")
fprintf("             SIMULATION RESULT             \n")
fprintf("===========================================\n\n")
   
err_matrix = Y_hist - YD_hist;                                    % CALCULATE RAW ERROR MATRIX

for y_idx = 1:size(err_matrix, 1)
    rms_val = sqrt(mean(err_matrix(y_idx, 1:t_idx).^2));          % RMS CALCULATION FOR EACH OUTPUT
    fprintf("RMS Error (y%d): %.4f\n", y_idx, rms_val);
end
fprintf("\n")

%% ==============================================================
%% FIGURE (1) — STATE AND REFERENCE
%% ==============================================================
figure(1); clf
num_y = size(Y_hist, 1);
tiledlayout(num_y, 1);

pos = get(gcf, "Position"); pos(4) = 420 * num_y/2; set(gcf, "Position", pos);

for y_idx = 1:num_y
    nexttile
    plot(t(1:t_idx), Y_hist(y_idx, 1:t_idx), 'blue', "LineWidth", line_width); hold on
    plot(t(1:t_idx), YD_hist(y_idx, 1:t_idx), 'green', "LineWidth", line_width, 'LineStyle', '--');
    ylabel("$y_{" + string(y_idx) + "}$", "FontSize", font_size, "Interpreter", "latex")
    xlabel("$t$ [s]", "FontSize", font_size, "Interpreter", "latex")
    grid on
    ax = gca; ax.FontSize = axes_font_size; ax.FontName = font_name;
    if y_idx == 1, legend("Actual", "Desired", "FontSize", 12); end
end

%% ==============================================================
%% FIGURE (2) — TRACKING ERROR
%% ==============================================================
figure(2); clf
num_yd = size(YD_hist, 1);
tiledlayout(num_yd, 1);

pos = get(gcf, "Position"); pos(4) = 420 * num_yd/2; set(gcf, "Position", pos);

for yd_idx = 1:num_yd
    nexttile
    plot(t(1:t_idx), YD_hist(yd_idx, 1:t_idx) - Y_hist(yd_idx, 1:t_idx), ...
        'red', "LineWidth", line_width); 
    ylabel("$e_{" + string(yd_idx) + "}$", "FontSize", font_size, "Interpreter", "latex")
    xlabel("$t$ [s]", "FontSize", font_size, "Interpreter", "latex")
    grid on
    ax = gca; ax.FontSize = axes_font_size; ax.FontName = font_name;
end

%% ==============================================================
%% FIGURE (3) — CONTROL INPUT
%% ==============================================================
figure(3); clf
num_u = size(U_hist, 1);
tiledlayout(num_u, 1);

pos = get(gcf, "Position"); pos(4) = 420 * num_u/2; set(gcf, "Position", pos);

for u_idx = 1:num_u
    nexttile
    plot(t(1:t_idx), U_hist(u_idx, 1:t_idx), 'black', "LineWidth", line_width); 
    ylabel("$u_{" + string(u_idx) + "}$", "FontSize", font_size, "Interpreter", "latex")
    xlabel("$t$ [s]", "FontSize", font_size, "Interpreter", "latex")
    grid on
    ax = gca; ax.FontSize = axes_font_size; ax.FontName = font_name;
end

%% ==============================================================
%% FIGURE (4) — WEIGHT NORM (CVL + FCL)
%% ==============================================================
figure(4); clf
pos = get(gcf, "Position"); pos(4) = 420; set(gcf, "Position", pos);
hold on

% CVL PLOTS
if NN.paramCtrl.CVLon
    for Om_idx = 1:NN.paramCtrl.CVL_num+1
        Om_Combined_norm = result.Om_hist.("Om_Combined"+string(Om_idx-1));
        plot(t(1:t_idx), Om_Combined_norm(1, 1:t_idx), ...
            'DisplayName', "$\Omega_{" + string(Om_idx-1) + "}$", ...
            "LineWidth", line_width);
    end
end

% FCL PLOTS
for V_idx = 1:1:NN.paramCtrl.FCL_num+1
    plot(t(1:t_idx), result.V_hist(V_idx, 1:t_idx), ...
        'DisplayName', "$V_{" + string(V_idx-1) + "}$", ...
        "LineWidth", line_width);
end

ylabel("Weight Norm", "FontSize", font_size, "Interpreter", "latex")
xlabel("$t$ [s]", "FontSize", font_size, "Interpreter", "latex")
grid on
lgd = legend; lgd.Location = "northeast"; lgd.NumColumns = 3; lgd.Interpreter = "latex";
ax = gca; ax.FontSize = axes_font_size; ax.FontName = font_name;

%% ==============================================================
%% FIGURE (5) — GLOBAL TRAJECTORY AND HEADING
%% ==============================================================
figure(5); clf
pos = get(gcf, "Position"); pos(3:4) = [800, 600]; set(gcf, "Position", pos);

% PLOT DESIRED REFERENCE PATH
plot(XY_pos_des(1, 1:t_idx), XY_pos_des(2, 1:t_idx), 'k--', 'LineWidth', 1.2, 'DisplayName', 'Desired Path'); 
hold on;

% PLOT ACTUAL VEHICLE PATH
plot(XY_pos(1, 1:t_idx), XY_pos(2, 1:t_idx), 'b', 'LineWidth', 2, 'DisplayName', 'Actual Path');

% % PLOT HEADING VECTORS (QUIVERS)
% skip = round(t_idx / 25);                                         % SHOW ~25 ORIENTATION ARROWS
% if skip < 1, skip = 1; end
% Psi_hist = Y_hist(2, :);                                          % GLOBAL HEADING DATA

% q = quiver(XY_pos(1, 1:skip:t_idx), XY_pos(2, 1:skip:t_idx), ...
%            cos(Psi_hist(1:skip:t_idx)), sin(Psi_hist(1:skip:t_idx)), ...
%            5, 'r', 'LineWidth', 1.5, 'DisplayName', 'Heading Vector'); % SCALE = 5
% q.MaxHeadSize = 0.5;

% FORMATTING
grid on; axis equal;
xlabel("$X$ [m]", "FontSize", font_size, "Interpreter", "latex")
ylabel("$Y$ [m]", "FontSize", font_size, "Interpreter", "latex")
title("Global Trajectory Tracking", "FontSize", font_size, "Interpreter", "latex")
legend("Location", "best", "FontSize", 12, "Interpreter", "latex");
ax = gca; ax.FontSize = axes_font_size; ax.FontName = font_name;

%% SAVE RESULT
if saveResult 
    result_dir = "result/" + string(exp_name);
    if ~exist(result_dir, 'dir'), mkdir(result_dir); end
    save(result_dir + "/result.mat", "result");

    for j = 1:length(figure_name)
        if saveFigFile
            saveas(figure(j), result_dir + "/" + figure_name(j) + ".fig")
        end
        saveas(figure(j), result_dir + "/" + figure_name(j) + ".png")
    end
end

end