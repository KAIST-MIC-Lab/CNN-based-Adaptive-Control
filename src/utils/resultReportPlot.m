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
];

font_size       = 24;
axes_font_size  = 18;
font_name       = "Times New Roman";
line_width      = 1.5;

%% PREPARE
t     = result.t;
t_idx = result.t_idx;
NN    = result.NN;

Y_hist = result.Y_hist;  
YD_hist = result.YD_hist;
U_hist  = result.U_hist;

%% ==============================================================
%% SIMULATION RESULT REPORT
%% ==============================================================
fprintf("===========================================\n")
fprintf("             SIMULATION RESULT             \n")
fprintf("===========================================\n\n")
   

err = Y_hist - YD_hist;
err = sqrt(sum(err.^2, 2));

for y_idx = 1:length(err)
    fprintf("RMS Error (x%d): %.4f\n", y_idx, err(y_idx));
end
fprintf("\n")

%% ==============================================================
%% FIGURE (1) — STATE AND REFERENCE
%% ==============================================================
figure(1); clf
num_y = size(Y_hist,1);

tiledlayout(num_y,1);

pos = get(gcf,"Position"); pos(4) = 420; set(gcf,"Position",pos);

for y_idx = 1:num_y
    nexttile
    plot(t(1:t_idx), Y_hist(y_idx,1:t_idx), 'blue', "LineWidth", line_width); hold on
    plot(t(1:t_idx), YD_hist(y_idx,1:t_idx), 'green', "LineWidth", line_width);
    ylabel("x_"+string(y_idx), "FontSize", font_size, "Interpreter","latex")
    xlabel("$t$ [s]", "FontSize", font_size, "Interpreter","latex")
    grid on
    ax = gca; ax.FontSize = axes_font_size; ax.FontName = font_name;
end

%% ==============================================================
%% FIGURE (2) — TRACKING ERROR
%% ==============================================================
figure(2); clf
num_yd = size(YD_hist, 1);

tiledlayout(num_yd,1);

pos = get(gcf,"Position"); pos(4) = 420; set(gcf,"Position",pos);

for yd_idx = 1:num_yd
    nexttile
    plot(t(1:t_idx), YD_hist(yd_idx,1:t_idx) - Y_hist(yd_idx,1:t_idx), ...
        'blue', "LineWidth", line_width); hold on
    
    ylabel("e_"+string(yd_idx), "FontSize", font_size, "Interpreter","latex")
    xlabel("$t$ [s]", "FontSize", font_size, "Interpreter","latex")
    grid on
    ax = gca; ax.FontSize = axes_font_size; ax.FontName = font_name;
end

%% ==============================================================
%% FIGURE (3) — CONTROL INPUT
%% ==============================================================
figure(3); clf
num_u = size(U_hist,1);

tiledlayout(num_u ,1);

pos = get(gcf,"Position"); pos(4) = 420; set(gcf,"Position",pos);

for u_idx = 1:num_u
    nexttile
    plot(t(1:t_idx), U_hist(u_idx,1:t_idx), 'blue', "LineWidth", line_width); hold on
    ylabel("u_"+string(u_idx), "FontSize", font_size, "Interpreter","latex")
    xlabel("$t$ [s]", "FontSize", font_size, "Interpreter","latex")
    grid on
    ax = gca; ax.FontSize = axes_font_size; ax.FontName = font_name;
end

%% ==============================================================
%% FIGURE (4) — WEIGHT NORM (CVL + FCL)
%% ==============================================================
figure(4); clf

pos = get(gcf,"Position"); pos(4) = 420; set(gcf,"Position",pos);

hold on

%% === CVL PLOTS ===
if NN.paramCtrl.CVLon
    for Om_idx = 1:NN.paramCtrl.CVL_num+1
        Om_Combined_norm = result.Om_hist.("Om_Combined"+string(Om_idx-1));
        plot(t(1:t_idx), Om_Combined_norm(1,1:t_idx), ...
            'DisplayName', "\Omega_"+string(Om_idx-1), ...
            "LineWidth", line_width);
    end
end

%% === FCL PLOTS ===
for V_idx = 1:1:NN.paramCtrl.FCL_num+1
    plot(t, result.V_hist(V_idx,:), ...
        'DisplayName', "V_"+string(V_idx-1), ...
        "LineWidth", line_width);
end

ylabel("Weight Norm", "FontSize", font_size, "Interpreter","latex")
xlabel("$t$ [s]", "FontSize", font_size, "Interpreter","latex")
grid on

lgd = legend; 
lgd.Location = "northeast";
lgd.NumColumns = 3;
lgd.FontSize = 15;

ax = gca; ax.FontSize = axes_font_size; ax.FontName = font_name;

%% SAVE RESULT
if saveResult 
    result_dir = "result/" + string(exp_name);

    if ~exist(result_dir, 'dir')
        mkdir(result_dir);
    end

    save(result_dir + "/result.mat", "result");

    for j = 1:length(figure_name)
        if saveFigFile
            saveas(figure(j), result_dir + "/" + figure_name(j) + ".fig")
        end
        saveas(figure(j), result_dir + "/" + figure_name(j) + ".png")
    end
end

end
