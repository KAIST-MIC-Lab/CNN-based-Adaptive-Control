function [Phi,lstm] = lstm_forward(ctrl_param, lstm, x)
    lstm_param = ctrl_param.lstm_param;             % lstm parameters
    bc= lstm_param.bc;                              % cell state decay rate
    bh= lstm_param.bh;                              % hidden state decay rate
    
    
    % 1. Construct z = [x; h; 1]
    z = [x; lstm.h; 1];                             % [l1 x 1]

    % 2. Compute gates 
    f_t = sigmoid(lstm.Wf' * z);                    % [l2 x 1]
    i_t = sigmoid(lstm.Wi' * z);                    % [l2 x 1]
    o_t = sigmoid(lstm.Wo' * z);                    % [l2 x 1]
    c_star = tanh(lstm.Wc' * z);                    % Candidate cell state c*

    % 3. Continuous-time States Integrator 
    % c integration
    psi_c = (f_t .* lstm.c) + (i_t .* c_star);
    dot_c = -bc * lstm.c + bc * psi_c;
    lstm.c = lstm.c + dot_c * ctrl_param.dt;

    % h integration
    psi_h = o_t .* tanh(lstm.c);
    dot_h = -bh * lstm.h + bh * psi_h;
    lstm.h = lstm.h + dot_h * ctrl_param.dt;
    
    % 4. Compute Control Output Phi
    Phi = lstm.Wh' * (psi_h + tanh(lstm.Wff' * x));

    % 5.Store for backpropagation
    lstm.z = z;
    lstm.psi_c = psi_c;
    lstm.psi_h = psi_h;
end

function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end