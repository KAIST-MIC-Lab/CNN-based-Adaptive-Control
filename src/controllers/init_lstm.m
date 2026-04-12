function lstm = init_lstm(lstm_param)
    % Extract dimensions
    input_size = ...                          % size of state x (Input size)
       lstm_param.input_size;      
    l2 = lstm_param.l2;                       % h, c size
    l1 = lstm_param.l1;                       % z size
    m  = lstm_param.m;                        % size of control output 

    % Xavier Scaling Factors
    sigma_gate = sqrt(2 / (l1 + l2));
    sigma_out  = sqrt(2 / (l2 + m));

    % Gate Weights [l1 x l2]
    lstm.Wf = randn(l1, l2) * sigma_gate;
    lstm.Wi = randn(l1, l2) * sigma_gate;
    lstm.Wo = randn(l1, l2) * sigma_gate;
    lstm.Wc = randn(l1, l2) * sigma_gate;

    % Output Weights
    lstm.Wh  = randn(l2, m) * sigma_out;       % [l2 x m]
    
    % Input state to control outputs directly
    lstm.Wff = randn(input_size, l2) * 0.01;    % [m x input_size]

    % Initial Internal States
    lstm.h = zeros(l2, 1);
    lstm.c = zeros(l2, 1);
    lstm.z = zeros(l1, 1); 
    lstm.f_t = zeros(l2, 1);
    lstm.i_t = zeros(l2, 1);
    lstm.o_t = zeros(l2, 1);
    lstm.c_star = zeros(l2, 1);
    lstm.psi_c = zeros(l2, 1);
    lstm.psi_h = zeros(l2, 1);
end
    
