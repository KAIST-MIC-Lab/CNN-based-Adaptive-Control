function NN = init_NN(NN)
    %% NEURAL NETWORK WEIGHTS INITIALIZATION

    init_range = 0.00001; % for now just one


    % CVL initialization
    if NN.paramCtrl.CVLon
        for nn_idx = 1:1:NN.paramCtrl.CVL_num+1
            n = NN.paramCtrl.CVL_Node(nn_idx, 2); % input width
            q = NN.paramCtrl.CVL_Node(nn_idx, 3); % filter height
            r = NN.paramCtrl.CVL_Node(nn_idx, 4); % filter number
    
            % uniformly initialized in [-init_range, init_range] (Xavier's Method)
    %         init_range = 1/(n*q);
            init_range = 0.1;
    
            % initialize
            % Omega is set of filters(W)
            % NN.("Omega"+string(nn_idx-1)) = rand(q,n,r) * init_range;
            NN.("Omega"+string(nn_idx-1)) = (rand(q,n,r) -0.5) * 2 * init_range;
            % NN.("Omega"+string(nn_idx-1)) = randn(q,n,r) * init_range;
            % Omega_B is bias of filter set Omega
            % NN.("Omega_B"+string(nn_idx-1)) = rand(r, 1) * init_range;
            NN.("Omega_B"+string(nn_idx-1)) = (rand(r, 1) -0.5) * 2 * init_range;
            % NN.("Omega_B"+string(nn_idx-1)) = randn(r, 1) * init_range;
    
            % calc weight numbers
            NN.paramCtrl.CVL_weight_num = NN.paramCtrl.CVL_weight_num + n*q*r + r;
        end
    end

    % LSTM initialization
    if NN.paramCtrl.LSTMon
        init_range = 0.1;
        
        h1 = NN.paramCtrl.LSTM_CS_size;
        h2 = NN.paramCtrl.LSTM_N_size;

        % gate weights initialization
        gate_name = ["c", "i", "f", "o"];
       for nn_idx = 1:1:length(gate_name)
            % NN.("W"+gate_name(nn_idx)) = rand(h1,h2) * init_range;
            NN.("W"+gate_name(nn_idx)) = (rand(h1,h2) - 0.5) * 2 * init_range;
            % NN.("W"+gate_name(nn_idx)) = randn(h1,h2) * init_range;
        end

        % hidden, cell states initialization
        NN.h = zeros(h2,1);
        NN.c = zeros(h2,1);

        % weights num calc
        NN.paramCtrl.LSTM_weight_num = h1*h2*4;
    end

    % FCL initialization
    for nn_idx = 1:1:NN.paramCtrl.FCL_num+1
        n = NN.paramCtrl.FCL_Node(nn_idx); 
        m = NN.paramCtrl.FCL_Node(nn_idx+1);
        
        % uniformly initialized in [-init_range, init_range] (Xavier's Method)
        % init_range = 1/(n*m);
        if nn_idx == NN.paramCtrl.FCL_num+1
            init_range = 0.01; % for now just one
        else
            init_range = 0.01; % for now just one
        end

        % initialize
        % since V includes bias term, row of V += 1
        % NN.("V"+string(nn_idx-1)) =  rand(n+1,m) * init_range;
        NN.("V"+string(nn_idx-1)) = (rand(n+1,m) -0.5) * 2 * init_range;
        % NN.("V"+string(nn_idx-1)) = randn(n+1,m) * init_range;
        
        % calc weight numbers
        NN.paramCtrl.FCL_weight_num = NN.paramCtrl.FCL_weight_num + n*m + n;
    end
end