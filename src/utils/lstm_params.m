function lstm_params = lstm_params()
    lstm_params.input_size = 2;                   % lstm input size
    lstm_params.l2 = 12;                          % hidden layer size
    lstm_params.l1 = lstm_params.input_size ... 
        + lstm_params.l2 + 1;                     % z= [x, h, 1] size
    lstm_params.bc = 5;                           % c integration constant
    lstm_params.bh = 1;                           % h integration constant
    lstm_params.n= 2;                             % Number of states
    lstm_params.m = 2;                            % Number of control outputs
end