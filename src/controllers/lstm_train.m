function lstm = lstm_train(ctrl_param, lstm, e, Phi)
    % 1. Extract dimensions and parameters
    eta = ctrl_param.eta;                           % learning rate
    Q   = ctrl_param.Q;                             % error weight matrix [2x2]
    R   = ctrl_param.R;                             % control effort weight
    lstm_param = ctrl_param.lstm_param;             % lstm parameters
    l2 = lstm_param.l2;                             % hidden layer size
    m   = size(lstm.Wh, 2);                         % number of control outputs

    % 2. Extract LSTM parameters
    % weights
    Wf = lstm.Wf;                                   % [l1 x l2]
    Wi = lstm.Wi;                                   % [l1 x l2]
    Wo = lstm.Wo;                                   % [l1 x l2]
    Wc = lstm.Wc;                                   % [l1 x l2]
    Wh = lstm.Wh;                                   % [l2 x m]
    Wff = lstm.Wff;                                 % [m x input_size]
    
    % internal states and gate activations
    c= lstm.c;                                      % [l2 x 1]
    z= lstm.z;                                      % [l1 x 1]
    psi_c = lstm.psi_c;                             % [l2 x 1]
    psi_h = lstm.psi_h;                             % [l2 x 1]

    % 3. Update vectorized Wh
    temp= (Q * e) + (R * Phi);  
    dPhi_dWh = kron(eye(m), (psi_h + tanh(Wff' * e))');
    Wh_vec = Wh(:) - eta * dPhi_dWh' * temp;
    lstm.Wh = reshape(Wh_vec, size(Wh));

    % 4. Wff update
    dPhi_dWff = Wh'*diag(tanh_derivative(Wff'*e))* ... % Vectorized Jacobian
                kron(eye(l2),e');                
    lstm.Wff = Wff(:) - eta * dPhi_dWff'*temp;
    lstm.Wff = reshape(lstm.Wff, size(Wff));

    % 5. Wf, Wi, Wo, Wc updates
    % dPsiC_d(vect(Wj)) for j in {f, i, o, c}
    dPsiC_dWc= diag(sigmoid(Wi'*z))*...
               diag(tanh_derivative(Wc'*z))*(kron(eye(l2),z'));
    dPsiC_dWi= diag(tanh(Wc'*z))*...
               diag(sigmoid_derivative(Wi'*z))*(kron(eye(l2),z'));
    dPsiC_dWf= diag(c)*diag(sigmoid_derivative(Wf'*z))*...
               kron(eye(l2),z');
    
    % dPsiH_dWj for j in {f, i, o, c}
    dPsiH_dWo= diag(tanh(psi_c))*...
               diag(sigmoid_derivative(Wo'*z))*(kron(eye(l2),z'));
    temp= diag(sigmoid(Wo'*z))*diag(tanh_derivative(psi_c));
    dPsiH_dWf= temp*dPsiC_dWf;
    dPsiH_dWi= temp*dPsiC_dWi;
    dPsiH_dWc= temp*dPsiC_dWc;

    % update Wf, Wi, Wo, Wc
    temp = Wh*((Q * e) + (R * Phi));  
    Wf_vec = Wf(:) - eta * dPsiH_dWf'*temp;
    Wi_vec = Wi(:) - eta * dPsiH_dWi'*temp;
    Wc_vec = Wc(:) - eta * dPsiH_dWc'*temp;
    Wo_vec = Wo(:) - eta * dPsiH_dWo'*temp;

    % Reshape back to original dimensions
    lstm.Wf = reshape(Wf_vec, size(lstm.Wf));
    lstm.Wi = reshape(Wi_vec, size(lstm.Wi));
    lstm.Wc = reshape(Wc_vec, size(lstm.Wc));
    lstm.Wo = reshape(Wo_vec, size(lstm.Wo));

end


%% Helper functions
function s = sigmoid(x)
    s = 1 ./ (1 + exp(-x));
end

function sd= sigmoid_derivative(x)
    s = 1 ./ (1 + exp(-x));
    sd = s .* (1 - s);
end

function td = tanh_derivative(x)
    t = tanh(x);
    td = 1 - t.^2;
end