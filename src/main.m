%% main.m
% ===========================================================================
%  CONAC-CVL CONTROL SIMULATION MAIN SCRIPT
%  Original Author: Myeongseok Ryu
%  Modified by: Naol Samuel
%  Last Modified: 2026-03-23
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
paramSim.T = 50;                                                     % TERMINAL TIME
t = 0:paramSim.dt:paramSim.T;
rpt_dt = 1;

% x = [e_y, e_y_dot, e_psi, e_psi_dot, theta_1, theta_2]
% u = [w_1, w_2] (Steering input rates)
x = zeros(6,1);                                                   % INITIAL STATE
u = zeros(2,1);                                                   % INITIAL CONTROL INPUT
y = [0; 0];                                                       % INITIAL OUTPUT (LATERAL AND HEADING) IN BODY FRAME
psi_des = 0;                                                      % INITIAL DESIRED HEADING
yd= [0; psi_des];                                                 % DESIRED OUTPUT (LATERAL Displacement= 0, HEADING = psi_des)

X_glob = 0;                                                       % INITIAL GLOBAL X COORDINATE
Y_glob = 0;                                                       % INITIAL GLOBAL Y COORDINATE




%% PASSIVE PARAMETERS
rng(paramSim.seed_num);
paramSim.exp_name = datetime('now','TimeZone','local', ...
    'Format','yyMMdd_HHmmss');


%% SYSTEM SPECIFICATIONS
Plant = paramPlant_load();

%% REFERENCE TRAJECTORY
ref_Traj = @(t) get_ref(t, Plant);                                  % REFERENCE TRAJECTORY FUNCTION HANDLE   


%% NEURAL NETWORK DECLARE
NN = paramCtrl_load(paramSim);
NN = init_NN(NN);


%% REPORT SIMULATION SETTING
reportSim(NN, paramSim);

%% RECORDER
recordPrepare                                                       % PREPARE TRAJECTORY RECORDERS
%% MAIN LOOP
stk_in= zeros( ...                                                  % BUFFER FROM WHICH CVL INPUTS ARE SAMPLED
    NN.paramCtrl.size_CVL_input(1)*int64(NN.paramCtrl.input_dt/NN.paramCtrl.dt), ...
    NN.paramCtrl.size_CVL_input(2));

fprintf("===========================================\n")
fprintf("             SIMULATION START              \n")
fprintf("===========================================\n")
fprintf("\n")
    
try 
    for t_idx = 2:1:length(t)
        %% 1. REFERENCE
        Ref = ref_Traj(t(t_idx));                                   % GET REFERENCE AT CURRENT TIME
        psi_des = psi_des + Ref.psi_dot_des * paramSim.dt;          % UPDATE DESIRED HEADING
        yd = [0; psi_des];                                                 % UPDATE DESIRED OUTPUT


        %% 2. NN INPUT PREPARATION
        e= [x(1); x(3)];                                            % ERROR SELECTION (LATERAL AND HEADING ERROR)
        nn_input= [e; u]/1000;                                      % CURRENT ERROR AND CONTROL INPUT AS NN INPUT
        
            
        %% 2. CONTROL LAW CALCULATION
        [NN_Out, NN, dataset_y] = NNforward(NN, nn_input, stk_in, t(t_idx));
        u = NN_Out
        
        %% 3. SYSTEM STEP
        x_dot = systemDynamics(x, u, Plant, Ref);    
        x = x + x_dot * paramSim.dt;

        y_dot=[ x(2) - Plant.vx*x(3); 
                x(4)]+ Ref.psi_dot_des;                           % OUTPUT DERIVATIVE CALCULATION

        y = y + y_dot * paramSim.dt;                              % OUTPUT UPDATE


        % GLOBAL POSITION UPDATE FOR VISUALIZATION
        beta = (x(2) + Plant.vx * x(3)) / Plant.vx;               % SIDESLIP ANGLE CALCULATION                                        % CURRENT GLOBAL HEADING
        X_dot_glob = Plant.vx * cos(y(2) + beta);            % GLOBAL X DOT WITH SIDESLIP
        Y_dot_glob = Plant.vx * sin(y(2) + beta);
        X_glob = X_glob + X_dot_glob * paramSim.dt;               % GLOBAL X UPDATE
        Y_glob = Y_glob + Y_dot_glob * paramSim.dt;
        
        %% 4. NEURAL NETWORK TRAINING
        NN = NNtrain(NN, e);
    
        %% 5. RECORDING
        result.Y_hist(:, t_idx)  = y;
        result.YD_hist(:, t_idx) = yd;  
        result.U_hist(:, t_idx)  = u;
        result.E_hist(:, t_idx)  = e;
        result.XY_pos(:, t_idx) = [X_glob; Y_glob];                 % RECORD GLOBAL POSITION FOR VISUALIZATION
        result.XY_pos_des(:, t_idx) = [Ref.X_des; Ref.Y_des];       % RECORD DESIRED GLOBAL POSITION FOR VISUALIZATION
       
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