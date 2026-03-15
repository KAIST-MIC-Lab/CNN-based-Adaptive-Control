%% main.m
% ===========================================================================
%  CONAC-CVL CONTROL SIMULATION MAIN SCRIPT
%  Original Author: Myeongseok Ryu
%  Modified by: Naol Samuel
%  Last Modified: 2025:12:08
% ===========================================================================   
%% 
clear;
clc;
close all;
addpath("utils")

%% SIMULATION SETTING
paramSim.saveResult  =   0;
paramSim.saveFigFile =   0;
paramSim.saveNetwork =   0;

% paramSim.seed_num =  1;
paramSim.seed_num =  130;

%% SIMULATION PARAMETERS
paramSim.dt = 1e-2;                                                 % SAMPLING TIME STEP
paramSim.T = 5;                                                     % TERMINAL TIME
t = 0:paramSim.dt:paramSim.T;
rpt_dt = 1;

x = [0; 0; 0; 0; 0; 0; 0];                                          % INITIAL STATE, in R^7
u = [0; 0];                                                         % Target steering angles
y=  [0; 0;];



%% PASSIVE PARAMETERS
rng(paramSim.seed_num);
paramSim.exp_name = datetime('now','TimeZone','local', ...
    'Format','yyMMdd_HHmmss');


%% SYSTEM SPECIFICATIONS
Plant = paramPlant_load();

%% REFERENCE
% Target states: [Y_ref; vy_ref; Psi_ref; r_ref]
ref_Traj = @(t) [ 
    2*sin(0.5*t);                                                  % Global Y path
    0.5*cos(0.5*t);                                                % Global Psi (Heading)
    ];    


%% NEURAL NETWORK DECLARE
NN = paramCtrl_load(paramSim);
NN = init_NN(NN);

%% REPORT SIMULATION SETTING
reportSim(NN, paramSim);

%% RECORDER
recordPrepare     
                                                                    % PREPARE TRAJECTORY RECORDERS
%% MAIN LOOP
dataset_y = zeros( ...                                              % BUFFER FROM WHICH CVL INPUTS ARE SAMPLED
    NN.paramCtrl.size_CVL_input(1)*int64(NN.paramCtrl.input_dt/NN.paramCtrl.dt), ...
    NN.paramCtrl.size_CVL_input(2));

fprintf("===========================================\n")
fprintf("             SIMULATION START              \n")
fprintf("===========================================\n")
fprintf("\n")
    
try 
    for t_idx = 2:1:length(t)
        %% 1. ERROR CALCULATION
        yd = ref_Traj(t(t_idx));
        y  = [x(1); x(3)];                            % Extract actual tracking output
        e  = y - yd;                            
    
        %% 2. CONTROL LAW CALCULATION
        % RECTIFIED: Pass 'y' (4x1) instead of 'x' (7x1) to match (10, 4) CVL input
        [NN_Out, NN, dataset_y] = NNforward(NN, y, yd, u, dataset_y, t(t_idx));
        u = -NN_Out;
        
        %% 3. SYSTEM STEP
        x_dot = systemDynamics(x, u, Plant);    % Use full 7-D physics
        x = x + x_dot * paramSim.dt;
        % y is updated at the start of the next loop iteration
        
        %% 4. NEURAL NETWORK TRAINING
        NN = NNtrain(NN, e);
    
        %% 5. RECORDING
        result.Y_hist(:, t_idx)  = y;
        result.YD_hist(:, t_idx) = yd;
        result.U_hist(:, t_idx)  = u;
        result.E_hist(:, t_idx)  = e;
       
        if NN.paramCtrl.CVLon                                       % RECORD WEIGHTS NORM OF CVL: Om and Om_B combined 
            for Om_idx = 1:1:NN.paramCtrl.CVL_num+1
                Om = NN.("Omega"+string(Om_idx-1));
                B = NN.("Omega_B"+string(Om_idx-1));
                Om_flat = Om(:);
                B_flat = B(:);
                OmB_flat = [Om_flat; B_flat];
                result.Om_hist.("Om_Combined"+string(Om_idx-1))(1, t_idx) = ...
                    norm(OmB_flat, "fro");
            end
        end
        
        for V_idx = 1:1:NN.paramCtrl.FCL_num+1                      % RECORD WEIGHTS NORM OF FCL: V
            result.V_hist(V_idx, t_idx) = norm(NN.("V"+string(V_idx-1)), "fro");
        end
       %% REPORTING
        if rem(t(t_idx)/paramSim.dt, rpt_dt/paramSim.dt) == 0
            fprintf("[INFO] Simulation Step %.2f/%.2fs (%.3f%%)\r", ...
                t(t(t_idx)/paramSim.dt) / paramSim.T, t(t_idx), paramSim.T, t(t_idx)/paramSim.T*100);
        end
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
    NN.gradTape = [];                                                 % CLEAR GRADIENT TAPE BEFORE SAVING
    save(result_dir + "/NN.mat","NN")
    fprintf("[INFO] Network Saved\n\n")
end

%% TERMINATION
fprintf("===========================================\n")
fprintf("           SIMULATION TERMINATED           \n")
fprintf("===========================================\n")
fprintf("\n")