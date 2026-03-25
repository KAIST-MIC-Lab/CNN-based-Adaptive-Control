function NN = paramCtrl_load(paramSim)
    %% NEURAL NETWORK PARAMETERS CONTROL LOAD
    NN.paramCtrl.CVLon = 1;                                         % CVL ON/OFF
    NN.paramCtrl.Gamma = 10;                                       % LEARNING RATE
    NN.paramCtrl.Beta = 1e-06;                                      % LAGRANGE MULTIPLIER LEARNING RATE (BETA_i, single value for simplicity)
    %NN.paramCtrl.initRangeCVL = 0.1;                               % CVL WEIGHTS INITIALIZATION RANGE(UNIFORM)
    %NN.paramCtrl.initRangeFCL = 0.1;                               % FCL WEIGHTS INITIALIZATION RANGE(UNIFORM)

    %% CONTROLLER'S PARAMETERS
    NN.paramCtrl.dt = paramSim.dt;                                  % SIMULATION TIME STEP 
    NN.paramCtrl.input_dt = 1e-1;                                   % CONTROLLER INPUT TIME STEP    

    %% NN SIZE
    NN.paramCtrl.size_CVL_input = [10, 4]; 
    NN.paramCtrl.size_FCL_input = 4;                                % IF CVL IS ON, THIS WILL BE OVERRIDED
    NN.paramCtrl.size_FCL_output = 2; 
            
    %% NEURAL NETWORK PARAMETERS AND OTHER SETTING
    NN.paramCtrl.FCL_phi = "tanh";                                  % FCL ACTIVATION FUNCTION
    NN.paramCtrl.CVL_phi = "tanh";                                  % CVL ACTIVATION FUNCTION

    %% CVL STRUCUTRE
    if NN.paramCtrl.CVLon
        NN.paramCtrl.CVL_filter_size = ...
            [ % q(filter height),     r(filter number)
            5       2
            3       2
            ];
        
        % CVL_Node MATRIX TO STORE DIMENSION OF EACH LAYER.
        % INCLUDING SIZE OF FILTERS AND NUMBER OF FILTERS
        NN.paramCtrl.CVL_Node = ...                                 % n(HEIGHT), m(WIDTH),q(FILTER HEIGHT),r(FILTER NUMBER)
        zeros(size(NN.paramCtrl.CVL_filter_size, 1)+1, 4);

        NN.paramCtrl.CVL_Node(1:end-1, 3:4) = ...                   % FILTERS SIZE
        NN.paramCtrl.CVL_filter_size;

        NN.paramCtrl.CVL_Node(1,1:2) = ...                          % CVL INPUT SIZE
        NN.paramCtrl.size_CVL_input;

        NN.paramCtrl.CVL_Node(2:end,2) = ...                        % CALCULATE WIDTHS OF EACH LAYER AFTER CONVOLUTION
        NN.paramCtrl.CVL_filter_size(1:end, 2);

        for CVL_idx = 2:1:size(NN.paramCtrl.CVL_filter_size, 1)+1
            NN.paramCtrl.CVL_Node(CVL_idx, 1) = ...                 % CALCULATE HEIGHTS OF EACH LAYER AFTER CONVOLUTION
                NN.paramCtrl.CVL_Node(CVL_idx-1, 1) - NN.paramCtrl.CVL_Node(CVL_idx-1, 3) + 1;
        end
    end

    %% FCL STRUCUTRE
    % MODIFY FCL INPUT SIZE IF CVL IS ON
    if NN.paramCtrl.CVLon
        NN.paramCtrl.size_FCL_input = ...
            NN.paramCtrl.CVL_Node(end,1)*NN.paramCtrl.CVL_Node(end,2);  
    end

    % FCL_Node MATRIX TO STORE DIMENSION OF EACH LAYER.
    NN.paramCtrl.FCL_Node = ...
        [
        NN.paramCtrl.size_FCL_input; 
        4
        NN.paramCtrl.size_FCL_output
        ];
        
    %% NORM CONTRAINT SETTINGS
    NN.paramCtrl.Om_norms=[                                          % WEIGHTS NORM CONSTRAINTS FOR CVL LAYERS
      100;
      100;       
    ];
    NN.paramCtrl.V_norms=[                                           % WEIGHTS NORM CONSTRAINTS FOR FCL LAYERS
      100;
      100;        
    ];
    NN.paramCtrl.NN_Out_norm = 10;                                   % NN OUTPUT NORM CONSTRAINT

    %% LAGRANGE MULTIPLIERS
    % Weight Constraint Multipliers (lambda_i)
    NN.paramCtrl.Lambda_Om = zeros(size(NN.paramCtrl.Om_norms));      % LAMBDA for CVL weights, initialized to zero
    NN.paramCtrl.Lambda_V = zeros(size(NN.paramCtrl.V_norms));        % LAMBDA for FCL weights, initialized to zero
    NN.paramCtrl.Lambda_Out = 0;                                     % LAMBDA for Output norm, initialized to zero
    
    %% PASSIVE PARAMETERS
    NN.paramCtrl.FCL_num = length(NN.paramCtrl.FCL_Node)-2;          % INDEX OF FCL WEIGHTS: 0 TO FCL_num
    if NN.paramCtrl.CVLon
        NN.paramCtrl.CVL_num = size(NN.paramCtrl.CVL_Node, 1)-2;     % INDEX OF CVL WEIGHTS: 0 TO CVL_num
    end
    NN.paramCtrl.FCL_weight_num = 0;
    NN.paramCtrl.CVL_weight_num = 0;
end