function [NN_Out, NN, dataset_y] = NNforward(NN, y, yd, u, dataset_y, t)
    %% PREPARE
    paramCtrl = NN.paramCtrl;
    CVLon = NN.paramCtrl.CVLon;

    %% NN INPUT CONSTRUCTION
    error = y-yd;
    nn_input = [error;u]/1000;                                  % NN INPUT CONSTRUCTION
    if isscalar(t)
        current_time = t;
        t_idx = round(current_time / NN.paramCtrl.dt) + 1;        % CALCULATE INDEX (must be >= 2)
    else 
        error('NNforward expects the current scalar time (t(t_idx)) as the last argument.')
    end

    if NN.paramCtrl.CVLon                                        
        dataset_y(1:end-1, :) = dataset_y(2:end, :);              % UPDATE BUFFER AT SIMULATION STEP
        dataset_y(end, :) = nn_input';
        
        % SLOW STACKING LOGIC
        NN_dt = NN.paramCtrl.dt;                                  % SIMULATION TIME STEP
        IN_dt = NN.paramCtrl.input_dt;                            % INPUT UPDATE TIME STEP
        IN_H = NN.paramCtrl.size_CVL_input(1);                    % CVL HISTORY HEIGHT (ROWS)
        
        UP_steps = round(IN_dt / NN_dt);                          % UPDATE PERIOD IN STEPS
        if rem(t_idx - 1, UP_steps) == 0 || t_idx == 2            % CHECK UPDATE CONDITION
                                                                 
            indices = int64(1:1:IN_H) * UP_steps;
            % Sample CVL INPUT and store it for persistence
            NN.paramCtrl.stk_x = dataset_y(indices, :);           % STACKED INPUT PERSISTENCE 
            
            % Check if the indices exceed the buffer size
            if max(indices) > size(dataset_y, 1)
                 error("CVL input history is larger than dataset_y buffer size. Increase buffer size in main script.")
            end
        end
        
        % Use the last stored stacked input 
        stacked_x = NN.paramCtrl.stk_x;                           % USE PERSISTENT STACKED INPUT 
    end
    disp("reached")
    lgn = 2;                                                      % LOGISTIC FUNCTION GAIN
    mx = 100;                                                     % SCALING FACTOR

    FCL_num = paramCtrl.FCL_num;
    %% CVL CALC
    if CVLon
        phi_CVL  = 2./(1+exp(-lgn * stacked_x)) - 1;              % LOGISTIC NORMALIZATION FOR CVL INPUTS   
        phi_CVL = mx*phi_CVL;                                     % SCALE INPUTS
        NN.gradTape.("CVL_phi0") = phi_CVL;                       % STORE INPUT LAYER ACTIVATION FOR GRADIENT TAPE         
    
        [phi, phi_dot] = phiSelect(paramCtrl.CVL_phi);
        for nn_idx = 0:1:paramCtrl.CVL_num
            % Weights and biases
            Om = NN.("Omega"+string(nn_idx));
            B = NN.("Omega_B"+string(nn_idx));

            % Forward pass
            CVL_out = CVL1D(phi_CVL, Om, B);
            phi_CVL = phi(CVL_out);

            %Store activations and derivatives 
            NN.gradTape.("CVL_phi"+string(nn_idx+1)) = phi_CVL;   % STORE ACTIVATIONS FOR GRADIENT TAPE
            if nn_idx ~= paramCtrl.CVL_num                        % NO NEED TO STORE DERIVATIVES AT OUTPUT LAYER
                NN.gradTape.("CVL_phi_dot"+string(nn_idx+1)) ...  % STORE DERIVATIVES FOR GRADIENT TAPE
                = phi_dot(CVL_out);
            end
        end
    end

    %% FCL CALC
    if CVLon 
        phi_FCL = [reshape(CVL_out, [], 1); 1];
    else % (FCL)
        nn_input  = 2./(1+exp(-lgn * nn_input)) - 1;
        nn_input = mx*nn_input;
        phi_FCL = [nn_input; 1];
    end

    [phi, phi_dot] = phiSelect(paramCtrl.FCL_phi);
    NN.gradTape.("FCL_phi0") = phi_FCL;
    for nn_idx = 0:1:FCL_num
        V = NN.("V"+string(nn_idx));
        FCL_out = V'*phi_FCL;
        phi_FCL = phi(FCL_out);                                   % JUNK AT OUTPUT LAYER
        phi_dot_FCL = phi_dot(FCL_out);
        if nn_idx == FCL_num 
            NN_Out = FCL_out;                                     % FINAL OUTPUT
            phi_FCL = FCL_out;                                    % STORE FINAL OUTPUT WITHOUT ACTIVATION
        else
            NN.gradTape.("FCL_phi"+string(nn_idx+1)) = phi_FCL;
            NN.gradTape.("FCL_phi_dot"+string(nn_idx+1)) = phi_dot_FCL;
        end
    end 
    NN.paramCtrl.NN_Out = NN_Out;                                 % USED IN NNTRAIN
    if length(NN_Out) ~= paramCtrl.size_FCL_output
        error("[ERR] NN output size is not same")
    end
end


%% ACTIVATION FUNCTIONS
function [phi, phi_dot] = phiSelect(phi)                          % SELECT ACTIVATION FUNCTION
    if phi == "tanh"
        phi = @(x) tanh_(x);
        phi_dot = @(x) tanh_dot(x);
    elseif phi == "relu"
        phi = @(x) relu_(x);
        phi_dot = @(x) relu_dot(x); 
    end
end

function x = tanh_(x)
    if isvector(x)
        x = [tanh(x); 1];
    else
        x = tanh(x);
    end
end

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

function x = relu_(x)
    if isvector(x)
        x = [max(0,x); 1];
        % x = [max(0.1*x,x); 1];
    else
        x = max(0,x);
        % x = max(0.1*x,x);
    end
end

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

