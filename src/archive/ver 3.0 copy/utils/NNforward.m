function [u, NN, dataset_x] = NNforward(NN, x, xd, u, dataset_x)
    %% PREPARE
    paramCtrl = NN.paramCtrl;
    
    CVLon = NN.paramCtrl.CVLon;
    LSTMon = NN.paramCtrl.LSTMon;

    error = x - xd;
    % ===============================================================
    error = error(2) + 1 * error(1);
    % ===============================================================
    dt = NN.paramCtrl.dt;

    % nn_input = [error;x;u];
    nn_input = [error;x;u]/1000;
    % nn_input = [
    %     diag([1/1,1/2])*error;
    %     diag([1/1,1/3])*x;
    %     diag([1/20,1/50])*u
    %     ];
    % nn_input = randn(size(nn_input)) * 0.1;
    % nn_input = zeros(size(nn_input));
    % nn_input = ones(size(nn_input)) * sin(u);

    if NN.paramCtrl.CVLon 
        if NN.paramCtrl.CVL2Don % 2D CVL
            stacked_x = (double(dataset_x)/255.0 - 0.5) * 2;
        else % 1D CVL
            dataset_x(1:end-1, :) = dataset_x(2:end, :);
            dataset_x(end, :) = nn_input';
    
            % stacked_x = flip(dataset_x);
            stacked_x = dataset_x( ...
            int64(1:1:NN.paramCtrl.size_CVL_input(1))*(NN.paramCtrl.input_dt/NN.paramCtrl.dt), :);
        end
    end

    lgn = 2;
    mx = 100;

    FCL_num = paramCtrl.FCL_num;


    %% CVL CALC
    if CVLon
        CVL_num = paramCtrl.CVL_num;

        % phi_CVL  = stacked_x;
        % phi_CVL  = tanh(stacked_x);

        phi_CVL  = 2./(1+exp(-lgn * stacked_x)) - 1;
        phi_CVL = mx*phi_CVL;
        % phi_CVL = double(stacked_x);

        NN.gradTape.("O_phi0") = phi_CVL;
    
        [phi, phi_dot] = phiSelect(paramCtrl.CVL_phi);
        for nn_idx = 1:1:paramCtrl.CVL_num+1
            Om = NN.("Omega"+string(nn_idx-1));
            B = NN.("Omega_B"+string(nn_idx-1));
    
            % CVL feedforwad
            CVL_out = CVL1D(phi_CVL, Om, B);
            phi_CVL = phi(CVL_out);
    
            % gradient tape
            NN.gradTape.("O_phi"+string(nn_idx)) = phi_CVL;
            if nn_idx ~= FCL_num+1
                NN.gradTape.("O_phi_dot"+string(nn_idx)) = phi_dot(CVL_out);
            end
        end
    end

    %% LSTM CALC
    if LSTMon
        if CVLon % CVL LSTM
            LSTM_x = reshape(CVL_out, [], 1); % concanate
        else % FCL LSTM
            nn_input  = 2./(1+exp(-lgn * nn_input)) - 1;
            LSTM_x = mx*nn_input;
        end
        z = [LSTM_x; NN.h; 1]; % 1 augmented for bias

        % gate calc
        f = tanh(NN.Wf' * z);
        i = tanh(NN.Wi' * z);
        c_star = sigmoid(NN.Wc' * z);
        o = tanh(NN.Wo' * z);
        
        % cell, hidden states gradients
        Psi_c = f .* NN.c + i .* c_star;
        Psi_h = o .* sigmoid(Psi_c);
        
        % gradient Tape
        NN.gradTape.z = z;
        NN.gradTape.c = NN.c;

        NN.gradTape.Psi_c_phi = sigmoid(Psi_c);
        NN.gradTape.Psi_c_phi_dot = sigmoid_dot(Psi_c);

        NN.gradTape.Psi_h = Psi_h;

        NN.gradTape.Wo_phi = o;
        NN.gradTape.Wo_phi_dot = ...
            eye(length(o)) - diag(o.^2);

        NN.gradTape.Wc_phi = c_star;
        NN.gradTape.Wc_phi_dot = sigmoid_dot(c_star);

        NN.gradTape.Wi_phi = i;
        NN.gradTape.Wi_phi_dot = ...
            eye(length(i)) - diag(i.^2);

        NN.gradTape.Wf_phi = f;
        NN.gradTape.Wf_phi_dot = ...
            eye(length(f)) - diag(f.^2);

        % cell, hiddenn states updates
        NN.c = NN.c + ...
            (-NN.paramCtrl.bc*NN.c + NN.paramCtrl.bc*Psi_c) * dt;
        NN.h = NN.h + ...
            (-NN.paramCtrl.bh*NN.h + NN.paramCtrl.bh*Psi_h) * dt;

        % output calc
        LSTM_out = Psi_h;
    end

    %% FCL CALC
    if LSTMon % (LSTM, LSTM+CVL)
        phi_FCL = [LSTM_out; 1];
    else
        if CVLon % (CVL)
            phi_FCL = [reshape(CVL_out, [], 1); 1];
        else % (FCL)
        nn_input  = 2./(1+exp(-lgn * nn_input)) - 1;
        nn_input = mx*nn_input;
        
        phi_FCL = [nn_input; 1];
        end
    end

    NN.gradTape.("V_phi0") = phi_FCL;

    [phi, phi_dot] = phiSelect(paramCtrl.FCL_phi);
    for nn_idx = 1:1:FCL_num+1
        V = NN.("V"+string(nn_idx-1));

        % FCL feedfowwad
        FCL_out = V'*phi_FCL;
        phi_FCL = phi(FCL_out);
        if nn_idx == FCL_num+1 
            % last layer does not need 1 augmentation and activation function
            phi_FCL = FCL_out;
        end

        % gradient tape
        if nn_idx ~= FCL_num+1
            NN.gradTape.("V_phi"+string(nn_idx)) = phi_FCL;
            NN.gradTape.("V_phi_dot"+string(nn_idx)) = phi_dot(FCL_out);  
        end
    end

    %% FINAL OUTPUT
    u = FCL_out;
    if length(u) ~= paramCtrl.size_FCL_output
        error("[ERR] NN output size is not same")
    end

end

%% LOCAL FUNCTIONS
function out = sigmoid(in)
    out = 1 / (1 + exp(-in));
    out = out';
end
function out = sigmoid_dot(in)
    out = diag(in.*(1-in));
end

% activation functino selector
function [phi, phi_dot] = phiSelect(phi)
    if phi == "tanh"
        phi = @(x) tanh_(x);
        phi_dot = @(x) tanh_dot(x);
    elseif phi == "relu"
        phi = @(x) relu_(x);
        phi_dot = @(x) relu_dot(x); 
    end
end

% tanh function
function x = tanh_(x)
    if isvector(x)
        x = [tanh(x); 1];
    else
        x = tanh(x);
    end
end

% derivative of tanh function
function x = tanh_dot(x)
    if isvector(x)
        x = [
            eye(length(x)) - diag(tanh(x).^2)
            zeros(1, length(x))
            ];
    else
        x = 1 - tanh(x).^2;
    end
end

% relu function
function x = relu_(x)
    if isvector(x)
        x = [max(0,x); 1];
        % x = [max(0.1*x,x); 1];
    else
        x = max(0,x);
        % x = max(0.1*x,x);
    end
end

% derivative of relu function
function x = relu_dot(x)
    if isvector(x)
        % x = [
        %     diag( ...
        %     sign(max(0,x)) + sign(min(0,x))*-0.1...
        %     )
        %     zeros(1,length(x))
        % ];

        x = [
            diag(sign(max(0, x)))
            zeros(1, length(x))
            ];
    else
        x = sign(max(0,x));
        % x = sign(max(0,x)) + sign(min(0,x))*-0.1;
    end
end



