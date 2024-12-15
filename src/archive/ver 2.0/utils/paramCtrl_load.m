function NN = paramCtrl_load(paramSim)
    %% Check! Check! Critical Setting
    % NN.paramCtrl.CVLon = 0;
    NN.paramCtrl.CVLon = 1;

    NN.paramCtrl.LSTMon = 0;
    % NN.paramCtrl.LSTMon= 1;

    NN.paramCtrl.RobustOn = 0;
    % NN.paramCtrl.RobustOn = 1;

    NN.paramCtrl.Dixon = 0; % E.K.
    % NN.paramCtrl.Dixon = 1; % Dixon

    NN.paramCtrl.Gamma = 1e+3; % learning rate
    % NN.paramCtrl.Gamma = 1e-1; % learning rate

    %% DIXON; no Ac, rho
    if NN.paramCtrl.Dixon
        NN.paramCtrl.rho = 0; % e-modification
        NN.paramCtrl.Ac = -eye(2);
    else
        % NN.paramCtrl.rho = NN.paramCtrl.Gamma*1e-1; % e-modification
        % NN.paramCtrl.rho = 10e-2; % e-modification
        % NN.paramCtrl.rho = 10e-1; % e-modification
        NN.paramCtrl.rho = 10e+0; % e-modification
        % NN.paramCtrl.rho = 50e+0; % e-modification
        NN.paramCtrl.Ac = diag([-10,-10]);
        % NN.paramCtrl.Ac = diag([-25,-25]);
        % NN.paramCtrl.Ac = diag([-50,-50]);
    end

    %% Controller's PARAMETERS
    % sampling time for controller
    NN.paramCtrl.dt = paramSim.dt;
    NN.paramCtrl.input_dt = 1e-1;
    % NN.paramCtrl.input_dt = 1e-4;
    % NN.paramCtrl.input_dt = 1e-2;

    % for global stability
    if NN.paramCtrl.RobustOn
        % NN.paramCtrl.ks = 0.2;
        NN.paramCtrl.ks = 2;
    else
        NN.paramCtrl.ks = 0;
    end

    NN.paramCtrl.inv_Ac = inv(NN.paramCtrl.Ac);

    %% NN SIZE
    NN.paramCtrl.size_CVL_input = [10, 6]; 
    % NN.paramCtrl.size_FCL_input = 6; % if CVL on, auto-determined
    NN.paramCtrl.size_FCL_input = NN.paramCtrl.size_CVL_input(end); % if CVL on, auto-determined
    NN.paramCtrl.size_FCL_output = 2; 
            
    %% PARAMETERS AND OTHER SETTING
    NN.paramCtrl.FCL_phi = "tanh";
    % NN.paramCtrl.FCL_phi = "relu";

    NN.paramCtrl.CVL_phi = "tanh";
    % NN.paramCtrl.CVL_phi = "relu";

    NN.paramCtrl.FCL_radius = 5e5;
    NN.paramCtrl.LSTM_radius = 5e2;
    NN.paramCtrl.CVL_radius = 5e2;

    %% CVL STRUCUTRE
    if NN.paramCtrl.CVLon
        NN.paramCtrl.CVL_filter_size = ...
            [
            % q(filter height),     r(filter number)
            2       4
            ];
        
        NN.paramCtrl.CVL_Node = zeros(size(NN.paramCtrl.CVL_filter_size, 1)+1, 4);
        % m(input height,    n(input width),    q(filter height),    r(filter number)
        % - input layer
        % - hidden layers
        % - output layer (q, r are is trash values; 0)
        NN.paramCtrl.CVL_Node(1:end-1, 3:4) = NN.paramCtrl.CVL_filter_size;
        NN.paramCtrl.CVL_Node(1,1:2) = NN.paramCtrl.size_CVL_input;
        NN.paramCtrl.CVL_Node(2:end,2) = NN.paramCtrl.CVL_filter_size(1:end, 2);
        for CVL_idx = 2:1:size(NN.paramCtrl.CVL_filter_size, 1)+1
            NN.paramCtrl.CVL_Node(CVL_idx, 1) = ...
                NN.paramCtrl.CVL_Node(CVL_idx-1, 1) - NN.paramCtrl.CVL_Node(CVL_idx-1, 3) + 1;
        end
    end

    %% LSTM STRUCTURE
    if NN.paramCtrl.LSTMon
        NN.paramCtrl.bc = 5e+1; % cell state converge gain
        NN.paramCtrl.bh = 5e+1; % cell state converge gain
      
        % cell, hidden state number (h2)
        NN.paramCtrl.LSTM_N_size = 4; 
        % concatenate state number (h1)
        if NN.paramCtrl.CVLon
            NN.paramCtrl.LSTM_CS_size = ...
                NN.paramCtrl.CVL_Node(end,1)*NN.paramCtrl.CVL_Node(end,2) ...
                + NN.paramCtrl.LSTM_N_size + 1;
        else
            NN.paramCtrl.LSTM_CS_size = ...
                NN.paramCtrl.size_FCL_input + NN.paramCtrl.LSTM_N_size + 1;
        end
        % LSTM input (h0)
        NN.paramCtrl.LSTM_in_size = ...
            NN.paramCtrl.LSTM_CS_size - NN.paramCtrl.LSTM_N_size - 1;
    end
    
    %% FCL STRUCUTRE
    NN.paramCtrl.FCL_Node = ...
        [
        NN.paramCtrl.size_FCL_input; % (FCL)
        8
        NN.paramCtrl.size_FCL_output
        ];  

    if NN.paramCtrl.LSTMon % (LSTM, CVL+LSTM)
        NN.paramCtrl.FCL_Node(1) = NN.paramCtrl.LSTM_N_size;
    elseif NN.paramCtrl.CVLon % (CVL)
        NN.paramCtrl.FCL_Node(1) = ...
            NN.paramCtrl.CVL_Node(end,1)*NN.paramCtrl.CVL_Node(end,2);  
    end

    %% PASSIVE PARAMETERS
    % ..._num means k_f, k_c index numbers
    % subtract 2 to exclude input/output layer
    % LSTM has single index
    NN.paramCtrl.FCL_num = length(NN.paramCtrl.FCL_Node)-2;

    if NN.paramCtrl.CVLon
        NN.paramCtrl.CVL_num = size(NN.paramCtrl.CVL_Node, 1)-2; 
    end


    NN.paramCtrl.FCL_weight_num = 0;
    NN.paramCtrl.CVL_weight_num = 0;
    NN.paramCtrl.LSTM_weight_num = 0;

    %% DEBUG
    % checkNNsize(NN.paramCtrl);


end

%% LOCAL FUNCTIONS
% function [] = checkNNsize(paramCtrl)
    % error("Check the size of CVL's output and input of FCL (must have same value)")
% end