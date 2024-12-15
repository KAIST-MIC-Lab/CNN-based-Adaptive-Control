function [u, NN, dataset_x] = NNforward(NN, x, xd, u, dataset_x)
    %% PREPARE
    paramCtrl = NN.paramCtrl;
    
    CNNon = NN.paramCtrl.CNNon;
    
    error = x - xd;

    nn_input = [error;x;u];
    dataset_x(1:end-1, :) = dataset_x(2:end, :);
    dataset_x(end, :) = nn_input';
    
    % stacked_x = flip(dataset_x);
    stacked_x = dataset_x( ...
        (1:1:NN.paramCtrl.size_CNN_input(1))*(NN.paramCtrl.input_dt/NN.paramCtrl.dt), :);
    
    %% PARAMETER LOAD
    CNN_num = paramCtrl.CNN_num;
    FCN_num = paramCtrl.FCN_num;

    %% CNN CALC
    if CNNon
        phi_CNN  = stacked_x;
        NN.gradTape.("O_phi0") = phi_CNN;
    
        [phi, phi_dot] = phiSelect(paramCtrl.CNN_phi);
        for nn_idx = 1:1:CNN_num+1
            Om = NN.("Omega"+string(nn_idx-1));
            B = NN.("Omega_B"+string(nn_idx-1));
    
            % CNN feedforwad
            CNN_out = CNN1D(phi_CNN, Om, B);
            phi_CNN = phi(CNN_out);
    
            % gradient tape
            NN.gradTape.("O_phi"+string(nn_idx)) = phi_CNN;
            if nn_idx ~= FCN_num+1
                NN.gradTape.("O_phi_dot"+string(nn_idx)) = phi_dot(CNN_out);
            end
        end
    end

    %% FCN CALC
    if CNNon
    % concanate output of CNN
        phi_FCN = [reshape(CNN_out, [], 1); 1]; % 1 augmented for bias
    else
        phi_FCN = [nn_input; 1];
    end

    NN.gradTape.("V_phi0") = phi_FCN;

    [phi, phi_dot] = phiSelect(paramCtrl.FCN_phi);
    for nn_idx = 1:1:FCN_num+1
        V = NN.("V"+string(nn_idx-1));

        % FCN feedfowwad
        FCN_out = V'*phi_FCN;
        phi_FCN = phi(FCN_out);
        if nn_idx == FCN_num+1 
            % last layer does not need 1 augmentation and activation function
            phi_FCN = FCN_out;
        end

        % gradient tape
        if nn_idx ~= FCN_num+1
            NN.gradTape.("V_phi"+string(nn_idx)) = phi_FCN;
            NN.gradTape.("V_phi_dot"+string(nn_idx)) = phi_dot(FCN_out);  
        end
    end

    %%
    u = FCN_out;
    if length(u) ~= paramCtrl.size_FCN_output
        error("[ERR] NN output size is not same")
    end

end

%% LOCAL FUNCTIONS
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
    else
        x = max(0,x);
    end
end

% derivative of relu function
function x = relu_dot(x)
    if isvector(x)
        x = [
            diag(sign(max(0, x)))
            zeros(1, length(x))
            ];
    else
        x = sign(max(0,x));
    end
end



