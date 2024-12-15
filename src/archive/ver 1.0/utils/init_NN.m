function NN = init_NN(NN)
    %% NEURAL NETWORK WEIGHTS INITIALIZATION
    % CNN initialization
    for nn_idx = 1:1:NN.paramCtrl.CNN_num+1
        n = NN.paramCtrl.CNN_Node(nn_idx, 2); % input width
        q = NN.paramCtrl.CNN_Node(nn_idx, 3); % filter height
        r = NN.paramCtrl.CNN_Node(nn_idx, 4); % filter number

        % uniformly initialized in [-init_range, init_range] (Xavier's Method)
%         init_range = 1/(n*q);
        init_range = 0.1; % for now just one

        % initialize
        % Omega is set of filters(W)
        NN.("Omega"+string(nn_idx-1)) = (rand(q,n,r) -0.5) * 2 * init_range;
        % Omega_B is bias of filter set Omega
        NN.("Omega_B"+string(nn_idx-1)) = (rand(r, 1) -0.5) * 2 * init_range;

        % calc weight numbers
        NN.paramCtrl.CNN_weight_num = NN.paramCtrl.CNN_weight_num + n*q*r + r;
    end

    % FCN initialization
    for nn_idx = 1:1:NN.paramCtrl.FCN_num+1
        n = NN.paramCtrl.FCN_Node(nn_idx); 
        m = NN.paramCtrl.FCN_Node(nn_idx+1);

        % uniformly initialized in [-init_range, init_range] (Xavier's Method)
%         init_range = 1/(n*m);
        init_range = 0.1; % for now just one

        % initialize
        % since V includes bias term, row of V += 1
        NN.("V"+string(nn_idx-1)) = (rand(n+1,m) -0.5) * 2 * init_range;
        
        % calc weight numbers
        NN.paramCtrl.FCN_weight_num = NN.paramCtrl.FCN_weight_num + n*m + n;
    end
end