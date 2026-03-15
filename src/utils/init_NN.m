function NN = init_NN(NN)
    %% NEURAL NETWORK WEIGHTS INITIALIZATION
    %NB:: XAVIER'S METHOD APPLIED HERE.
    %% CVL INITIALIZATION (XAVIER)
    if NN.paramCtrl.CVLon
        for nn_idx = 1:NN.paramCtrl.CVL_num+1

            n = NN.paramCtrl.CVL_Node(nn_idx, 2);                            % INPUT WIDTH
            q = NN.paramCtrl.CVL_Node(nn_idx, 3);                            % FILTER HEIGHT
            r = NN.paramCtrl.CVL_Node(nn_idx, 4);                            % FILTER NUMBER

            fan_in  = n * q;
            fan_out = r * q;

            lim = sqrt(6 / (fan_in + fan_out));

            % FILTERS
            NN.("Omega"+string(nn_idx-1)) = ...
                (rand(q,n,r)*2 - 1) * lim;

            % BIASES 
            NN.("Omega_B"+string(nn_idx-1)) = zeros(r,1);

            NN.paramCtrl.CVL_weight_num = ...
                NN.paramCtrl.CVL_weight_num + n*q*r + r;
        end
    end


    %% FCL INITIALIZATION (XAVIER)
    for nn_idx = 1:NN.paramCtrl.FCL_num+1

        n = NN.paramCtrl.FCL_Node(nn_idx);
        m = NN.paramCtrl.FCL_Node(nn_idx+1);

        fan_in  = n;
        fan_out = m;

        lim = sqrt(6 / (fan_in + fan_out));

        V = (rand(n+1,m)*2 - 1) * lim;

        % Bias row scaled smaller 
        V(end,:) = 0;

        NN.("V"+string(nn_idx-1)) = V;

        NN.paramCtrl.FCL_weight_num = ...
            NN.paramCtrl.FCL_weight_num + n*m + m;
    end

end