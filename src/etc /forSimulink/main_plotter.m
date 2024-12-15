clear

%% PLOTTER SETTING
selected = [1];
saveFigFile = 0;

%% DATA DIRECTORY
rst_dir = [
    % 1 FCN (Dixon) 10s
    "240219_220744";
    % 2 FCN (EK) 10s
    "240219_220843";
    % 3 FCN (EK, high damp) 10s
    "240219_221002";
    % 4 FCN (EK, high Ac) 10s
    "240219_221036";
    % 5 CNN (Dixon) 10s  
    "240219_222050";
    % 6 CNN (EK) 10s
    "240219_221604";
    % 7 CNN (EK, high damp) 10s
    "240219_222229";
    % 8 CNN (EK, high Ac) 10s
    "240219_222309";
    % 9 CNN (Dixon) 10s  (0.01 Ts)
    "240219_221136";
    % 10 CNN (EK) 10s  (0.01 Ts)
    "240219_222154";
    % 11 CNN (EK) 5s (system change)
    "240224_191230";
    % 12 FCN (EK) 5s (system change)
    "240224_191313";
    ];
%% DATA NAME
rst_name = [
    "FCN (Dixon)"
    "FCN (Proposed)"
    "FCN (Proposed; High rho)"
    "FCN (Proposed; High Ac)"
    "CNN (Dixon)"
    "CNN (Proposed)"
    "CNN (Proposed; High rho)"
    "FCN (Proposed; High Ac)"
    "CNN (Dixon; small Ts)"
    "CNN (Proposed; small Ts)"
    "CNN (Proposed; system change)"
    "CNN (Proposed; system change)"
];

%% DATA LOAD
fprintf("loading results...\n")
for rst_idx = 1:1:length(rst_name)
    rst_set.("rst"+string(rst_idx)) = load("result/"+rst_dir(rst_idx)+"/result.mat");
end

%% ERROR CALC
rst_list = [6 10 5 7 8 2];
ERR_list = zeros(2, length(rst_list));


for rst_idx = 1:1:length(rst_list)
    trg_idx = rst_list(rst_idx);
    result = rst_set.("rst"+string(trg_idx)).result;
    
    t = result.t;
    tf_idx = find(t==1.5);

    X_hist = result.X_hist(:, 1:tf_idx);
    XD_hist = result.XD_hist(:, 1:tf_idx);
    
    ERR = X_hist - XD_hist;
    ERR = ERR.^2;
    ERR = sum(ERR, 2);

    ERR_list(:, rst_idx) = ERR / 1e4;

end

disp(ERR_list)

%% PLOT PREPARE
fprintf("Plotting...\n")
figure_name = [
    "Fig1"
    "Fig2"
    "Fig3"
    "Fig4"
    "Fig5"
    "Fig6"
    "Fig7"
    "Fig8"
];

font_size = 24;
axes_font_size = 24;
font_name = "Times New Roman";
line_width = 3.5;

%% FIGURE 1~6
rst_list = [6 10 5 7 8 2];
title_list = [
    "CNN1" %CNN (EK)"
    "CNN2" %CNN (EK, Small stacking time)"
    "CNN3" %CNN (Dixon)"
    "CNN4" %CNN (EK, High rho)"
    "CNN5" %CNN (EK, High Ac)"
    "DNN" %FCN"
    ];

% ====================================================================================
% 6 10 5 7 8 2 CNN (EK)
for rst_idx = 1:1:length(rst_list)
    figure(rst_idx); clf


    gcf_tl = gcf;
    gcf_tl.Position(end) = 150;
    gcf_tl.Position(3) = 300;

    tl = tiledlayout(1,1);
    % tl.TileSpacing = 'compact';
    tl.Padding = 'compact';

    nexttile
    
    trg_idx = rst_list(rst_idx);
    result = rst_set.("rst"+string(trg_idx)).result;
    
    t = result.t;
    tf_idx = find(t==1.5);

    t = t(1:tf_idx);
    X_hist = result.X_hist(:, 1:tf_idx);
    XD_hist = result.XD_hist(:, 1:tf_idx);
    
    plot(t, X_hist(2,:)-XD_hist(2,:), 'red', "LineWidth", line_width, "LineStyle", ":"); hold on
    plot(t, X_hist(1,:)-XD_hist(1,:), 'blue', "LineWidth", line_width); hold on
    ylabel("$e_1,e_2$", "FontSize", font_size, "Interpreter","latex")
    xlabel("$t$ [s]", "FontSize", font_size, "Interpreter","latex")
    % ylim([min(min(X_hist(1,:),XD_hist(1,:))) *1.1, max(max(X_hist(1,:),XD_hist(1,:)) *1.1)])
    ylim([-1.5 3.5])
    grid on
    gca_tl = gca;
    gca_tl.FontSize = axes_font_size;
    gca_tl.FontName = font_name;
    gca_tl.XTick = 0:0.5:2;
    % gca_tl.YTick = -1.5:1.5:3.5;
    gca_tl.YTick = [-1.5,0,1.5,3];


    % plot(t, X_hist(1,:), 'blue', "LineWidth", line_width); hold on
    % plot(t, XD_hist(1,:), 'green', "LineWidth", line_width, "LineStyle", ":"); hold on
    % ylabel("$x_1$", "FontSize", font_size, "Interpreter","latex")
    % xlabel("$t$ [s]", "FontSize", font_size, "Interpreter","latex")
    % ylim([min(min(X_hist(1,:),XD_hist(1,:))) *1.1, max(max(X_hist(1,:),XD_hist(1,:)) *1.1)])
    % grid on
    % gca_tl = gca;
    % gca_tl.FontSize = axes_font_size;
    % gca_tl.FontName = font_name;
    % gca_tl.XTick = 0:0.5:2;
    % 
    % nexttile
    % plot(t, X_hist(2,:), 'blue', "LineWidth", line_width); hold on
    % plot(t, XD_hist(2,:), 'green', "LineWidth", line_width, "LineStyle", ":"); hold on
    % ylabel("$x_2$", "FontSize", font_size, "Interpreter","latex")
    % xlabel("$t$ [s]", "FontSize", font_size, "Interpreter","latex")
    % ylim([min(min(X_hist(2,:),XD_hist(2,:)) *1.1), max(max(X_hist(2,:),XD_hist(2,:)) *1.1)])
    % grid on
    % gca_tl = gca;
    % gca_tl.FontSize = axes_font_size;
    % gca_tl.FontName = font_name;
    % gca_tl.XTick = 0:0.5:2;

    % tl.Title.String = title_list(rst_idx);
    % tl.Title.FontName = font_name;
    % tl.Title.FontSize = font_size;
end

%% Fig 7
figure(7);clf

trg_idx = 6;
result = rst_set.("rst"+string(trg_idx)).result;

t = result.t;
tf_idx = find(t==2);

t = t(1:tf_idx);
NN = result.NN;
Om_hist = result.Om_hist;
V_hist = result.V_hist;

gcf_tl = gcf;
gcf_tl.Position(end) = 420;

if NN.paramCtrl.CNNon
    for Om_idx = 1:1:NN.paramCtrl.CNN_num+1
        Om_norm = Om_hist.("Om"+string(Om_idx-1));
        B_norm = Om_hist.("Om_B"+string(Om_idx-1));
        for filter_idx = 1:1:size(Om_norm, 1)
            plot(t, Om_norm(filter_idx, 1:tf_idx), ...
                'DisplayName', "\Omega_"+string(Om_idx-1)+":W_"+string(filter_idx) ...
                , "LineWidth", line_width); hold on
        end
        plot(t, B_norm(1, 1:tf_idx), 'DisplayName', "B_"+string(Om_idx-1) ...
            , "LineWidth", line_width); hold on
    end
end

for V_idx = 1:1:NN.paramCtrl.FCN_num
    plot(t, V_hist(V_idx, 1:tf_idx), 'DisplayName', "V_"+string(V_idx-1) ...
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

%% Fig 8
figure(8); clf
tl = tiledlayout(1,2);
% tl.TileSpacing = 'compact';
tl.Padding = 'compact';

gcf_tl = gcf;
gcf_tl.Position(end) = 150;    

rst_list = [11; 12];

rst_color = ["blue", "red"];

for rst_idx = 1:1:length(rst_list)


    nexttile(1)
    
    trg_idx = rst_list(rst_idx);
    result = rst_set.("rst"+string(trg_idx)).result;
    
    c = rst_color(rst_idx);

    t = result.t;
    ti_idx = find(t==2.5);
    tf_idx = find(t==4);

    t = t(ti_idx:tf_idx);
    X_hist = result.X_hist(:, ti_idx:tf_idx);
    XD_hist = result.XD_hist(:, ti_idx:tf_idx);
    
    plot(t, X_hist(1,:), "color", c, "LineWidth", line_width); hold on
    plot(t, XD_hist(1,:), 'green', "LineWidth", line_width, "LineStyle", ":"); hold on
    ylabel("$x_1$", "FontSize", font_size, "Interpreter","latex")
    xlabel("$t$ [s]", "FontSize", font_size, "Interpreter","latex")
    ylim([min(min(X_hist(1,:),XD_hist(1,:))) *1.1, max(max(X_hist(1,:),XD_hist(1,:)) *1.1)])
    grid on
    gca_tl = gca;
    gca_tl.FontSize = axes_font_size;
    gca_tl.FontName = font_name;
    gca_tl.XTick = 2.5:0.5:4;
    gca_tl.YTick = -1:1:1;

    nexttile(2)
    plot(t, X_hist(2,:), c, "LineWidth", line_width); hold on
    plot(t, XD_hist(2,:), 'green', "LineWidth", line_width, "LineStyle", ":"); hold on
    ylabel("$x_2$", "FontSize", font_size, "Interpreter","latex")
    xlabel("$t$ [s]", "FontSize", font_size, "Interpreter","latex")
    ylim([min(min(X_hist(2,:),XD_hist(2,:)) *1.1), max(max(X_hist(2,:),XD_hist(2,:)) *1.1)])
    grid on
    gca_tl = gca;
    gca_tl.FontSize = axes_font_size;
    gca_tl.FontName = font_name;
    gca_tl.XTick = 2.5:0.5:4;
    gca_tl.YTick = 0:0.4:2;

    % tl.Title.String = title_list(rst_idx);
    % tl.Title.FontName = font_name;
    % tl.Title.FontSize = font_size;
    

end

%% SAVE
fprintf("Saving...\n")

% for j = 1:1:length(figure_name)
% for j = selected

for j = 1:1:length(figure_name)
    plt = figure(j);
    exportgraphics(plt, "fig/" + figure_name(j) +'.eps')

    if saveFigFile
    saveas(figure(j), ...
        "fig/" + figure_name(j) + ".fig")
    end

    saveas(figure(j), ...
        "fig/"  + figure_name(j) + ".png")        
end


fprintf("Saved!\n")
