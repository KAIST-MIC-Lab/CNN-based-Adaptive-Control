function NN = NNtrain(NN, error)
    %% PREPARE
    % ===============================================================
    error = error(2) + 1 * error(1);
    % ===============================================================

    paramCtrl = NN.paramCtrl;
    gradTape = NN.gradTape;
    
    CVLon = paramCtrl.CVLon;
    LSTMon = paramCtrl.LSTMon;

    dt = paramCtrl.dt;

    out_num = paramCtrl.size_FCL_output;

    % CVL_num = paramCtrl.CVL_num;

    FCL_num = paramCtrl.FCL_num;
    FCL_Node = paramCtrl.FCL_Node;
    % FCL_weight_num = paramCtrl.FCL_weight_num;
    FCL_radius = paramCtrl.FCL_radius;
    
    if CVLon
        CVL_num = paramCtrl.CVL_num;
        CVL_Node = paramCtrl.CVL_Node;
        % CVL_weight_num = paramCtrl.CVL_weight_num;
        CVL_radius = paramCtrl.CVL_radius;
    end

    if LSTMon

    end

    Gamma = paramCtrl.Gamma;        % learning rate

    % =========================
    % from E.K.
    % employed Ac design matrix, e-modification
    inv_Ac = paramCtrl.inv_Ac;
    rho = paramCtrl.rho;
    % =========================


    %% BACK-PROPAGATION
    % FCL train ====================================================
    dPhidphi = 1;
    for nn_idx = flip(1:1:FCL_num+1)
        % for certain FCL layer
        phi = gradTape.("V_phi"+string(nn_idx-1));
        grad_ori = kron(eye(FCL_Node(nn_idx+1)), phi');
        if nn_idx ~= FCL_num+1
            V = NN.("V"+string(nn_idx));
            phi_dot = gradTape.("V_phi_dot"+string(nn_idx));

            % grad_ori = (V'*phi_dot) * grad_ori;
            dPhidphi = dPhidphi * (V'*phi_dot);

        end
        grad_ori = dPhidphi * grad_ori;

        V = NN.("V"+string(nn_idx-1));
        [n,m] = size(V);

        vecV = reshape(V, [], 1);

        % FCL update
        % grad = Gamma * grad_ori' * error;
        grad1 = -Gamma * grad_ori' * inv_Ac  * error;
        % grad1 = -Gamma * grad_ori' * [0 1] * inv_Ac  * error;
        grad2 = -rho * norm(error) * vecV;

        grad = grad1 + grad2;      
        grad = projGrad(grad, FCL_radius, vecV, Gamma, dt);
        % if sum(isinf(grad))
        %     warning("FCL gradients are inf")
        %     grad = zeros(size(grad));
        % end
        vecV = vecV + grad * dt;

        NN.("V"+string(nn_idx-1)) = reshape(vecV, n,m);
    end

    % back-propagation to next
    % if k_f = 2
    % grad_bckprop = (V2' * phi'2 * V1' * phi'1 * V0')'
    grad_bckprop = dPhidphi * NN.V0';
    grad_bckprop = grad_bckprop(:,1:end-1)'; % delete last gradient (from bias)

    % LSTM train ===================================================
    if LSTMon
        z = gradTape.z;
        l1 = paramCtrl.LSTM_CS_size; % concatenate stata size
        l2 = paramCtrl.LSTM_N_size; % cell, hidden state size
        n = paramCtrl.LSTM_in_size; % LSTM input size
        c = gradTape.c;

        % input -> Psi_c
        dPsicdWc = diag(gradTape.Wi_phi) ...
            * gradTape.Wc_phi_dot * kron(eye(l2), z');
        dPsicdWi = diag(gradTape.Wc_phi) ...
            * gradTape.Wi_phi_dot * kron(eye(l2), z');
        dPsicdWf = diag(c) * gradTape.Wf_phi_dot * kron(eye(l2), z');
        
        % Psi_c -> Psi_h
        dPsihdWc = diag(gradTape.Wo_phi) ...
            * gradTape.Wc_phi_dot * dPsicdWc;
        dPsihdWi = diag(gradTape.Wo_phi) ...
            * gradTape.Wc_phi_dot * dPsicdWi;
        dPsihdWf = diag(gradTape.Wo_phi) ...
            * gradTape.Wc_phi_dot * dPsicdWf;        
        dPsihdWo = diag(gradTape.Psi_c_phi) ...
            * gradTape.Wo_phi_dot * kron(eye(l2), z');
        
        % Phi_h -> Phi
        dPhidWc = grad_bckprop' * dPsihdWc;
        dPhidWi = grad_bckprop' * dPsihdWi;
        dPhidWf = grad_bckprop' * dPsihdWf;
        dPhidWo = grad_bckprop' * dPsihdWo;

        % back-propagation to next
        mapper = [eye(n); zeros(l2+1, n)];

        dfdx = gradTape.Wf_phi_dot * NN.Wf' * mapper;
        dc_stardx = gradTape.Wc_phi_dot * NN.Wc' * mapper;
        didx = gradTape.Wi_phi_dot * NN.Wi' * mapper;

        grad_bckprop = grad_bckprop' * diag(gradTape.Wo_phi) ...
            * gradTape.Wc_phi_dot ...
            * ( ...
                + diag(gradTape.c) * dfdx ...
                + diag(gradTape.Wi_phi) * dc_stardx ...
                + diag(gradTape.Wc_phi) * didx ...
            );
        grad_bckprop = grad_bckprop';
        
        % final gradients calc
        vecWc = reshape(NN.Wc, [], 1);
        grad1 = - Gamma * dPhidWc' * inv_Ac * error;
        grad2 = - rho * norm(error) * vecWc;
        vecWc = vecWc + (grad1 + grad2) * dt;

        vecWi = reshape(NN.Wi, [], 1);
        grad1 = - Gamma * dPhidWi' * inv_Ac * error;
        grad2 = - rho * norm(error) * vecWi;
        vecWi = vecWi + (grad1 + grad2) * dt;

        vecWf = reshape(NN.Wf, [], 1);
        grad1 = - Gamma * dPhidWf' * inv_Ac * error;
        grad2 = - rho * norm(error) * vecWf;
        vecWf = vecWf + (grad1 + grad2) * dt;   

        vecWo = reshape(NN.Wo, [], 1);
        grad1 = - Gamma * dPhidWo' * inv_Ac * error;
        grad2 = - rho * norm(error) * vecWo;
        vecWo = vecWo + (grad1 + grad2) * dt;

        % weights updates
        NN.Wc = reshape(vecWc, l1, l2);
        NN.Wi = reshape(vecWi, l1, l2);
        NN.Wf = reshape(vecWf, l1, l2);
        NN.Wo = reshape(vecWo, l1, l2);

    end

    % CVL train ====================================================
    if CVLon
        for out_idx = 1:1:out_num
            dPhidO = grad_bckprop(:, out_idx);
            dPhidO = reshape(dPhidO, CVL_Node(end,1), CVL_Node(end,2));
    
            for nn_idx = flip(1:1:CVL_num+1)
                % prepare
                phi = gradTape.("O_phi"+string(nn_idx-1));
                Om = NN.("Omega"+string(nn_idx-1));
                B = NN.("Omega_B"+string(nn_idx-1));
        
                filter_num = CVL_Node(nn_idx, end);
                filter_size = size(Om);
                phi_size = size(phi);
        
                % main gradient calc
                dPhidOm = zeros(size(Om));
                dPhidphi = zeros(phi_size);
                dPhidB = zeros(filter_num,1);
                for filter_idx = 1:1:filter_num
                    for row_idx = 1:1:filter_size(1)
                        % dPhi/dO -> dPhi/dOmega
                        dPhidOm(:,:,filter_idx) = ...
                            dPhidOm(:,:,filter_idx) + ...
                            dPhidO(row_idx,filter_idx) * phi(row_idx:row_idx+filter_size(1)-1, :);
                        % dPhi/dO -> dPhi/dB
                        dPhidB(filter_idx) = sum(dPhidO(:,filter_idx));
        
                        % dPhi/dO -> dPhi/dphi
                        tmp = zeros(phi_size);
                        tmp(row_idx:row_idx+filter_size(1)-1,:) = Om(:,:,filter_idx);
                        dPhidphi = dPhidO(row_idx,filter_idx) * tmp;
                    end
                end
        
                if nn_idx ~= 1
                    % dPhidphi -> dPhi/dO
                    phi_dot = gradTape.("O_phi_dot"+string(nn_idx-1));
                    dPhidO = dPhidphi .* phi_dot;
                end
        
                % CVL updta
                % Om_grad1 = Gamma * dPhidOm * error(1);
                % B_grad1 = Gamma * dPhidB * error(1);
                Om_grad1 = -Gamma * dPhidOm * inv_Ac(out_idx, out_idx) * error(1);
                B_grad1 =  -Gamma * dPhidB * inv_Ac(out_idx, out_idx)  * error(1);
                
                Om_grad2 = - rho * norm(error) * Om;
                B_grad2 = - rho * norm(error) * B;

                Om_grad = Om_grad1 + Om_grad2;
                B_grad = B_grad1 + B_grad2;

                % projection
                if sum(isinf(Om_grad))
                    warning("CVL (filter) gradients are inf")
                    Om_grad = zeros(size(Om_grad));
                end
                if sum(isinf(B_grad))
                    warning("CVL (bias) gradients are inf")
                    B_grad = zeros(size(B_grad));
                end

                Om = Om + Om_grad * dt;
                B = B + B_grad * dt;
        
                NN.("Omega"+string(nn_idx-1)) = Om;
                NN.("Omega_B"+string(nn_idx-1)) = B;
            end
        end
    end
end

%% LOCAL FUNCTIONS
function grad = projGrad(grad_V, radius, vecV, Gamma, dt)
    
    next_V = vecV + grad_V*dt;
    norm_nextV = norm(next_V);
    
    if sum(isnan(next_V))
        grad = zeros(size(grad_V));
    elseif norm_nextV <= radius
        grad = grad_V;
    else
        grad = 1/dt*(-vecV+next_V/norm_nextV*radius);
    end

    % % Gamma = eye(length(vecV)); % parameter in projection
    % epsilon = 10; % for smooth projection
    % 
    % grad_f = 1/2*(vecV'*vecV)^(-1/2)*2*vecV;
    % 
    % next_V = vecV + grad_V*dt;
    % 
    % if norm(next_V) <= radius
    %     grad = grad_V;
    % % originally projection operator only work on boundary
    % % however considering the discrete numerical computation
    % % relaxed the condition
    % else
    %     grad = eye(length(vecV)) - Gamma*(grad_f*grad_f')/(grad_f'*Gamma*grad_f);
    % 
    %     grad = grad*grad_V;
    % end
end


