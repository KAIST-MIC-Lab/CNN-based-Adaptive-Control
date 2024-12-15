% ============================================================
%       * Stable Online End-to-End CNN Controller *
%
%   Crafted By - Ryu Myeongseok
%   Version 1.0
%  
%   * General
%       - Relu is not working well
%       - Projection does not work
%   * CNN
%       - 1D-CNN
%       - Concatanate Layer
%       - No pooling layer
%   * DNN
%
% ============================================================

%% 
clear;
% clc;

addpath("utils")

%% SIMULATION SETTING
paramSim.saveResult  =   0;
paramSim.saveFigFile =   0;
paramSim.saveNetwork =   0;

paramSim.seed_num =  182;

%% SIMULATION PARAMETERS
paramSim.dt = 1e-4;                          % sampling time
paramSim.T = 5;                             % Termination time
% paramSim.T = 2;                             % Termination time
t = 0:paramSim.dt:paramSim.T;
rpt_dt = 1;

x = [1 2]';                     % initial state
u = [0 0]';                              % initial input

%% PASSIVE PARAMETERS
rng(paramSim.seed_num);

paramSim.exp_name = datetime('now','TimeZone','local', ...
    'Format','yyMMdd_HHmmss');

%% NEURAL NETWORK DECLARE
NN = paramCtrl_load(paramSim);
NN = init_NN(NN);

reportSim(NN, paramSim);
dataset_x = zeros( ...
    NN.paramCtrl.size_CNN_input(1)*ceil(NN.paramCtrl.input_dt/NN.paramCtrl.dt), ...
    NN.paramCtrl.size_CNN_input(2));

%% MAIN LOOP

fprintf("===========================================\n")
fprintf("             SIMULATION START              \n")
fprintf("===========================================\n")
fprintf("\n")
    
%%
sim("untitled1.slx");

%% PLOT AND REPORT
result.t = t;
result.NN = NN;

fprintf("[INFO] Plotting Results\n\n")
resultReportPlot(result, paramSim);

%% DIARY OFF
if paramSim.saveResult
    diary off
end

%% NETWORK SAVE
if paramSim.saveResult && paramSim.saveNetwork
    NN.gradTape = []; % clear gradTape
    save(result_dir + "/NN.mat","NN")
    fprintf("[INFO] Network Saved\n\n")
end

%% TERMINATION
fprintf("===========================================\n")
fprintf("           SIMULATION TERMINATED           \n")
fprintf("===========================================\n")
fprintf("\n")

