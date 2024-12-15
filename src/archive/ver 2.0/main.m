% ============================================================
%       * Stable Online End-to-End CNN Controller *
%
%   Crafted By - Ryu Myeongseok
%   Version 2.0
%  
%   * General
%       - [Target] LSTM add!
%       - Relu is not working well
%       - Projection does not work
%       - Layer's names are changed to FCL and CVL
%   * CNN
%       - 1D-CNN
%       - Concatanate Layer
%       - No pooling layer
%   * DNN
%
% ============================================================
% 
%% 
clear;
% clc;

addpath("utils")

%% SIMULATION SETTING
paramSim.saveResult  =   0;
paramSim.saveFigFile =   0;
paramSim.saveNetwork =   0;

% paramSim.seed_num =  1;
paramSim.seed_num =  130;

%% SIMULATION PARAMETERS
paramSim.dt = 1e-4;                          % sampling time
paramSim.T = 5;                             % Termination timeamSim.T = 1;                             % Termination time
t = 0:paramSim.dt:paramSim.T;
rpt_dt = 1;

x = [1 2]';                     % initial state
u = [0 0]';                              % initial input

%% SYSTEM DECLARE

% x1' = x2
% x2' = -5x1^3 -2x2 + u

% grad_x = @(x,u)[x(2); -5*x(1)^3-2*x(2)+u];

% from paper of Dixon
% grad_x = @(x,u, t) [
%     x(1)*x(2)*tanh(x(2))+sech(x(1)) + u(1)
%     sech(x(1)+x(2))^2-sech(x(2))^2 + u(2)
%     ];

grad_x = @(x,u, t) [
    % sin(x(1))*10 + x(2)^2*2*tanh(x(1)) + x(1)*x(2)*tanh(x(2))+sech(x(1)) + u(1) % +  2*heaviside(t-3)*(x(1)^2*x(2)+(sin(t)+10))
    % sin(x(2)+x(1))*10 + x(1)^2*2*sec(x(1)) + sech(x(1)+x(2))^2-sech(x(2))^2 + u(2) %+ 2*heaviside(t-3)*(x(2)^2*tanh(x(1))+(cos(1/2*t)+10))
    x(1)*x(2)*tanh(x(2))+sech(x(1)) + 1*heaviside(t-3)*(x(1)^2*x(2)+(sin(t)+10)) + u(1)
    sech(x(1)+x(2))^2-sech(x(2))^2 + 1*heaviside(t-3)*(x(2)^2*tanh(x(1))+(cos(1/2*t)+10)) + u(2)
    % -10*x(1) + u(1)
    % -20*x(2) + u(2)    
    ];

%% PASSIVE PARAMETERS
rng(paramSim.seed_num);

paramSim.exp_name = datetime('now','TimeZone','local', ...
    'Format','yyMMdd_HHmmss');

%% REFERENCE
ref_Traj = @(t)[ % desired trajectory
    sin(2*t)-cos(1.5*t)+1
    -cos(1*t)+1*sin(1.5*t)+1
    ] * 1e0;    

% ref_Traj = @(t)[ % desired trajectory
%     sin(2*t)
%     -cos(t)
%     ];    

%% NEURAL NETWORK DECLARE
NN = paramCtrl_load(paramSim);
NN = init_NN(NN);

reportSim(NN, paramSim);

%% RECORDER
% prepare trajectories recorder
recordPrepare

%% MAIN LOOP
dataset_x = zeros( ...
    NN.paramCtrl.size_CVL_input(1)*int64(NN.paramCtrl.input_dt/NN.paramCtrl.dt), ...
    NN.paramCtrl.size_CVL_input(2));

fprintf("===========================================\n")
fprintf("             SIMULATION START              \n")
fprintf("===========================================\n")
fprintf("\n")
    
try 
    for t_idx = 2:1:length(t)
        % ==============================================================
        % reference, error calc
        xd = ref_Traj(t(t_idx));
    
        % ==============================================================
        % control input
        [u_NN, NN, dataset_x] = NNforward(NN, x, xd, u, dataset_x);
        u = -1 * u_NN - NN.paramCtrl.ks * sign(x-xd);
        
        % max_u = 1e2;
        % u = min(max(u, -max_u), max_u);
        % ==============================================================
        % dynamics step
        x = x + grad_x(x, u, t(t_idx)) * paramSim.dt;
    
        % if abs(x(1)) > 2
        %     disp(x)
        % end
        % ==============================================================
        % train
        NN = NNtrain(NN, x-xd);
    
        % ==============================================================
        % history record
        result.X_hist(:, t_idx) = x;
        result.XD_hist(:, t_idx) = xd;
        result.U_hist(:, t_idx) = u;
        if NN.paramCtrl.CVLon
            for Om_idx = 1:1:NN.paramCtrl.CVL_num+1
                for filter_idx = 1:1:NN.paramCtrl.CVL_Node(Om_idx, end)
                    result.Om_hist.("Om"+string(Om_idx-1))(filter_idx, t_idx) = ...
                        norm(NN.("Omega"+string(Om_idx-1))(:,:,filter_idx), "fro");
                end    
                result.Om_hist.("Om_B"+string(Om_idx-1))(1,t_idx) = ...
                    norm(NN.("Omega_B"+string(Om_idx-1)));
            end
        end
        if NN.paramCtrl.LSTMon
            gate_name = ["c", "i", "f", "o"];
            for nn_idx = 1:1:length(gate_name)
                gt_name = gate_name(nn_idx);        
                result.("W"+gt_name+"_hist")(t_idx) = norm(NN.("W"+gt_name), "fro");
            end
        end
        for V_idx = 1:1:NN.paramCtrl.FCL_num
            result.V_hist(V_idx, t_idx) = norm(NN.("V"+string(V_idx-1)), "fro");
        end
    
        % ==============================================================
        % simulation report
        if rem(t(t_idx)/paramSim.dt, rpt_dt/paramSim.dt) == 0
            fprintf("[INFO] Simulation Step %.2f/%.2fs (%.3f%%)\r", ...
                t(t_idx), paramSim.T, t(t_idx)/paramSim.T*100);
        end
    
        % ==============================================================
        if isnan(x(1))
            error("states Inf")
        end
    end
catch whyStop
    fprintf("[ERROR] %s\n", whyStop.message)

    if paramSim.saveResult
        diary off
    end
    result.t = t;
    result.t_idx = t_idx;
    result.NN = NN;
    
    resultReportPlot(result, paramSim);
    return
end
%% PLOT AND REPORT
result.t = t;
result.t_idx = t_idx;
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

