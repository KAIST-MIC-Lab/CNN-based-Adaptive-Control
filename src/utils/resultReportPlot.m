function [] = resultReportPlot(result, paramSim)
%% SAVE?
saveResult = paramSim.saveResult;
saveFigFile = paramSim.saveFigFile;
exp_name = paramSim.exp_name;

%% PLOT SETTING
figure_name = [
    "state and ref"
    "error"
    "controls"
    "weight norm"
];

font_size = 24;
axes_font_size = 18;
font_name = "Times New Roman";
line_width = 1.5;

%% PREPARE
t = result.t;
t_idx = result.t_idx;
NN = result.NN;

X_hist = result.X_hist;  
XD_hist = result.XD_hist;
U_hist = result.U_hist;
if NN.paramCtrl.CVLon
    for Om_idx = 1:1:NN.paramCtrl.CVL_num+1
        Om_hist.("Om"+string(Om_idx-1)) = result.Om_hist.("Om"+string(Om_idx-1));
        Om_hist.("Om_B"+string(Om_idx-1)) = result.Om_hist.("Om_B"+string(Om_idx-1));
    end
end
if NN.paramCtrl.LSTMon
    gate_name = ["c", "i", "f", "o"];
    for nn_idx = 1:1:length(gate_name)
        gt_name = gate_name(nn_idx);        
        W_hist.("W"+gt_name+"_hist") = result.("W"+gt_name+"_hist");
    end

end
V_hist = result.V_hist;

%% RESULT REPORT
fprintf("===========================================\n")
fprintf("             SIMULATION RESULT             \n")
fprintf("===========================================\n")
fprintf("\n")
    
err = X_hist - XD_hist;
err = err.^2;
err = sum(err, 2);
err = sqrt(err);

for x_idx = 1:1:length(err)
    fprintf("RMS Error (x%d): %.3f\n", x_idx, err(x_idx))
end
fprintf("\n")

%% MAIN PLOT
% ==============================================================
% FIGURE(1) STATE 
figure(1); clf
num_x = size(X_hist,1);

tl = tiledlayout(num_x,1);

gcf_tl = gcf;
gcf_tl.Position(end) = 420;

for x_idx = 1:1:num_x
    nexttile
    plot(t(1:t_idx), X_hist(x_idx,1:t_idx), 'blue', "LineWidth", line_width); hold on
    plot(t(1:t_idx), XD_hist(x_idx,1:t_idx), 'green', "LineWidth", line_width); hold on
    ylabel("$x_1$", "FontSize", font_size, "Interpreter","latex")
    xlabel("$t$ [s]", "FontSize", font_size, "Interpreter","latex")
    grid on
    gca_tl = gca;
    gca_tl.FontSize = axes_font_size;
    gca_tl.FontName = font_name;
    ylim([-2.5 2.5])
end

% ==============================================================
% FIGURE(2) ERROR
figure(2); clf
num_xd = size(XD_hist, 1);

tl = tiledlayout(num_xd,1);

gcf_tl = gcf;
gcf_tl.Position(end) = 420;

for xd_idx = 1:1:num_xd
    nexttile
    plot(t(1:t_idx), (XD_hist(xd_idx,1:t_idx) - X_hist(xd_idx,1:t_idx)), 'blue', "LineWidth", line_width); hold on
    ylabel("$x^*_1 - x_1$", "FontSize", font_size, "Interpreter","latex")
    xlabel("$t$ [s]", "FontSize", font_size, "Interpreter","latex")
    grid on
    gca_tl = gca;
    gca_tl.FontSize = axes_font_size;
    gca_tl.FontName = font_name;
end

% ==============================================================
% FIGURE(3) INPUT
figure(3); clf
num_u = size(U_hist,1);

tl = tiledlayout(num_u ,1);

gcf_tl = gcf;
gcf_tl.Position(end) = 420;

for u_idx = 1:1:num_u 
    nexttile 
    plot(t(1:t_idx), U_hist(u_idx,1:t_idx), 'blue', "LineWidth", line_width); hold on
    ylabel("$u_1$", "FontSize", font_size, "Interpreter","latex")
    xlabel("$t$ [s]", "FontSize", font_size, "Interpreter","latex")
    grid on
    gca_tl = gca;
    gca_tl.FontSize = axes_font_size;
    gca_tl.FontName = font_name;
end

% ==============================================================
% FIGURE(4) network weight Frobenius norm
figure(4); clf

gcf_tl = gcf;
gcf_tl.Position(end) = 420;

if NN.paramCtrl.CVLon
    for Om_idx = 1:1:NN.paramCtrl.CVL_num+1
        Om_norm = Om_hist.("Om"+string(Om_idx-1));
        B_norm = Om_hist.("Om_B"+string(Om_idx-1));
        for filter_idx = 1:1:size(Om_norm, 1)
            plot(t, Om_norm(filter_idx,:), ...
                'DisplayName', "\Omega_"+string(Om_idx-1)+":W_"+string(filter_idx) ...
                , "LineWidth", line_width); hold on
        end
        plot(t, B_norm, 'DisplayName', "B_"+string(Om_idx-1) ...
            , "LineWidth", line_width); hold on
    end
end

if NN.paramCtrl.LSTMon
   gate_name = ["c", "i", "f", "o"];
    for nn_idx = 1:1:length(gate_name)
        gt_name = gate_name(nn_idx);
        w_hist = W_hist.("W"+gt_name+"_hist");
        plot(t, w_hist, ...
                'DisplayName', "W_"+gt_name ...
                , "LineWidth", line_width); hold on
    end
end

for V_idx = 1:1:NN.paramCtrl.FCL_num
    plot(t, V_hist(V_idx,:), 'DisplayName', "V_"+string(V_idx-1) ...
        , "LineWidth", line_width); hold on
end
ylabel("Weight Nrom", "FontSize", font_size, "Interpreter","latex")
xlabel("$t$ [s]", "FontSize", font_size, "Interpreter","latex")
grid on
lgd = legend;
lgd.Location = "northeast";
% lgd.Layout([3,[]])
lgd.NumColumns = 4;
lgd.FontSize = 15;
gca_tl = gca;
gca_tl.FontSize = axes_font_size;
gca_tl.FontName = font_name;

%% SAVE RESULT
if saveResult 
    result_dir = "result/" + string(exp_name);

    if ~exist(result_dir, 'dir')
       mkdir(result_dir);
    end

    save(result_dir + "/" + "result.mat", "result");

    for j = 1:1:length(figure_name)
        if saveFigFile
        saveas(figure(j), ...
            result_dir + "/" + figure_name(j) + ".fig")
        end
        saveas(figure(j), ...
            result_dir + "/" + figure_name(j) + ".png")        
    end
end

end