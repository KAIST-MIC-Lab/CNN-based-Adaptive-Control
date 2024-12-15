function NN = NNtrain(NN, error)
    %% PREPARE
    paramCtrl = NN.paramCtrl;
    gradTape = NN.gradTape;
    
    CNNon = paramCtrl.CNNon;

    dt = paramCtrl.dt;

    out_num = paramCtrl.size_FCN_output;

    % CNN_num = paramCtrl.CNN_num;

    FCN_num = paramCtrl.FCN_num;
    FCN_Node = paramCtrl.FCN_Node;
    % FCN_weight_num = paramCtrl.FCN_weight_num;
    FCN_radius = paramCtrl.FCN_radius;

    CNN_num = paramCtrl.CNN_num;
    CNN_Node = paramCtrl.CNN_Node;
    % CNN_weight_num = paramCtrl.CNN_weight_num;
    CNN_radius = paramCtrl.CNN_radius;
    
    Gamma = paramCtrl.Gamma;        % learning rate

    % =========================
    % from E.K.
    % employed Ac design matrix, e-modification
    inv_Ac = paramCtrl.inv_Ac;
    rho = paramCtrl.rho;
    % =========================


    %% BACK-PROPAGATION
    % FCN train ====================================================
    dPhidphi = 1;
    for nn_idx = flip(1:1:FCN_num+1)
        % for certain FCN layer
        phi = gradTape.("V_phi"+string(nn_idx-1));
        grad_ori = kron(eye(FCN_Node(nn_idx+1)), phi');
        if nn_idx ~= FCN_num+1
            V = NN.("V"+string(nn_idx));
            phi_dot = gradTape.("V_phi_dot"+string(nn_idx));

            % grad_ori = (V'*phi_dot) * grad_ori;
            dPhidphi = dPhidphi * (V'*phi_dot);

        end
        grad_ori = dPhidphi * grad_ori;

        V = NN.("V"+string(nn_idx-1));
        [n,m] = size(V);

        vecV = reshape(V, [], 1);

        % FCN update
        % grad = Gamma * grad_ori' * error;
        grad1 = -Gamma * grad_ori' * inv_Ac * error;
        grad2 = -rho * norm(error) * vecV;

        grad = grad1 + grad2;      
%         grad = projGrad(grad, FCN_radius, vecV, Gamma);
        vecV = vecV + grad * dt;

        NN.("V"+string(nn_idx-1)) = reshape(vecV, n,m);
    end

    % CNN train ====================================================
    if CNNon
        % back-propagation from FCN to CNN 
        % if k_f = 2
        % grad_bckprop = (V2' * phi'2 * V1' * phi'1 * V0')'
        grad_bckprop = dPhidphi * NN.V0';
        grad_bckprop = grad_bckprop(:,1:end-1)'; % delete last gradient (from bias)
        
        for out_idx = 1:1:out_num
            dPhidO = grad_bckprop(:, out_idx);
            dPhidO = reshape(dPhidO, CNN_Node(end,1), CNN_Node(end,2));
    
            for nn_idx = flip(1:1:CNN_num+1)
                % prepare
                phi = gradTape.("O_phi"+string(nn_idx-1));
                Om = NN.("Omega"+string(nn_idx-1));
                B = NN.("Omega_B"+string(nn_idx-1));
        
                filter_num = CNN_Node(nn_idx, end);
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
        
                % CNN updta
                % Om_grad1 = Gamma * dPhidOm * error(1);
                % B_grad1 = Gamma * dPhidB * error(1);
                Om_grad1 = - Gamma * dPhidOm * inv_Ac(out_idx, out_idx) * error(1);
                B_grad1 = - Gamma * dPhidB * inv_Ac(out_idx, out_idx)  * error(1);
                
                Om_grad2 = - rho * norm(error) * Om;
                B_grad2 = - rho * norm(error) * B;

                Om_grad = Om_grad1 + Om_grad2;
                B_grad = B_grad1 + B_grad2;

                % projection
                Om = Om + Om_grad * dt;
                B = B + B_grad * dt;
        
                NN.("Omega"+string(nn_idx-1)) = Om;
                NN.("Omega_B"+string(nn_idx-1)) = B;
            end
        end
    end
end

%% LOCAL FUNCTIONS
function grad = projGrad(grad_V, radius, vecV, Gamma)
    
%     Gamma = eye(length(vecV)); % parameter in projection
    epsilon = 10; % for smooth projection

    grad_f = 1/2*(vecV'*vecV)^(-1/2)*2*vecV;

    normV = norm(vecV);

    if normV-radius < 0 || grad_f'*grad_V <= 0
        grad = grad_V;
    % originally projection operator only work on boundary
    % however considering the discrete numerical computation
    % relaxed the condition
    elseif normV-radius >= 0 && grad_f'*grad_V > 0
        c = min(1, (normV-radius)/epsilon);
        grad = eye(length(vecV)) - c*Gamma*(grad_f*grad_f')/(grad_f'*Gamma*grad_f);
        grad = grad*grad_V;
    end

end

function grad = CNN_bckprop(W, bck_grad)
      
    filter_length = size(W,1) - size(bck_grad,1) + 1;

    grad = zeros(filter_length, size(W,2));

    for rep_idx = 1:1:size(bck_grad,1)
        grad = grad + bck_grad(rep_idx) * W(rep_idx:rep_idx+filter_length-1,:);
    end

end

%% BACK UP

% FCN UPDATE; FOR ...
%     for nn_idx = flip(1:1:FCN_num+1)
%         % for certain FCN layer
%         phi = gradTape.("V_phi"+string(nn_idx-1));
%         grad_ori = kron(eye(FCN_Node(nn_idx+1)), phi');
%         for trn_idx = nn_idx:1:FCN_num
%             V = NN.("V"+string(trn_idx));
%             phi_dot = gradTape.("V_phi_dot"+string(trn_idx));
% 
%             grad_ori = (V'*phi_dot) * grad_ori;
%         end
% 
%         V = NN.("V"+string(nn_idx-1));
%         [n,m] = size(V);
% 
%         vecV = reshape(V, [], 1);
% 
%         % FCN update
%         grad = Gamma * grad_ori' * error;
% %         grad = projGrad(grad, FCN_radius, vecV, Gamma);
%         vecV = vecV + grad * dt;
% 
%         NN.("V"+string(nn_idx-1)) = reshape(vecV, n,m);
%     end
% 
% % back-propagation from FCN to CNN ==============================
% grad_bckprop = NN.("V"+string(0))';
% for bckprop_idx = 1:1:FCN_num
%     V = NN.("V"+string(bckprop_idx));
%     phi_dot = gradTape.("V_phi_dot"+string(bckprop_idx));
% 
%     % if k_f = 2
%     % grad_bckprop = (V2' * phi'2 * V1' * phi'1 * V0')'
%     grad_bckprop = (V'*phi_dot) * grad_bckprop;
% end
% grad_bckprop = grad_bckprop(:,1:end-1)'; % row vec to col vec
% % grad_bckprop = reshape(grad_bckprop, CNN_Node(end,1), CNN_Node(end,2));

  
% RECURSIVE
% % FCN train ====================================================
%     dPhidphi = 1;
%     for nn_idx = flip(1:1:FCN_num+1)
%         % for certain FCN layer
%         phi = gradTape.("V_phi"+string(nn_idx-1));
%         grad_ori = kron(eye(FCN_Node(nn_idx+1)), phi');
%         if nn_idx ~= FCN_num+1
%             V = NN.("V"+string(nn_idx));
%             phi_dot = gradTape.("V_phi_dot"+string(nn_idx));
% 
%             % grad_ori = (V'*phi_dot) * grad_ori;
%             dPhidphi = dPhidphi * (V'*phi_dot);
% 
%         end
%         grad_ori = dPhidphi * grad_ori;
% 
%         V = NN.("V"+string(nn_idx-1));
%         [n,m] = size(V);
% 
%         vecV = reshape(V, [], 1);
% 
%         % FCN update
%         grad = Gamma * grad_ori' * error;
% %         grad = projGrad(grad, FCN_radius, vecV, Gamma);
%         vecV = vecV + grad * dt;
% 
%         NN.("V"+string(nn_idx-1)) = reshape(vecV, n,m);
%     end
% 
%     % CNN train ====================================================
%     if CNNon
%         % back-propagation from FCN to CNN 
%         % if k_f = 2
%         % grad_bckprop = (V2' * phi'2 * V1' * phi'1 * V0')'
%         grad_bckprop = dPhidphi * NN.V0';
%         grad_bckprop = grad_bckprop(:,1:end-1)'; % delete last gradient (from bias)
% 
%         for out_idx = 1:1:out_num
%             dPhidO = grad_bckprop(:, out_idx);
%             dPhidO = reshape(dPhidO, CNN_Node(end,1), CNN_Node(end,2));
% 
%             for nn_idx = flip(1:1:CNN_num+1)
%                 % prepare
%                 phi = gradTape.("O_phi"+string(nn_idx-1));
%                 Om = NN.("Omega"+string(nn_idx-1));
%                 B = NN.("Omega_B"+string(nn_idx-1));
% 
%                 filter_num = CNN_Node(nn_idx, end);
%                 filter_size = size(Om);
%                 phi_size = size(phi);
% 
%                 % main gradient calc
%                 dPhidOm = zeros(size(Om));
%                 dPhidphi = zeros(phi_size);
%                 dPhidB = zeros(filter_num,1);
%                 for filter_idx = 1:1:filter_num
%                     for row_idx = 1:1:filter_size(1)
%                         % dPhi/dO -> dPhi/dOmega
%                         dPhidOm(:,:,filter_idx) = ...
%                             dPhidOm(:,:,filter_idx) + ...
%                             dPhidO(row_idx,filter_idx) * phi(row_idx:row_idx+filter_size(1)-1, :);
%                         % dPhi/dO -> dPhi/dB
%                         dPhidB(filter_idx) = sum(dPhidO(:,filter_idx));
% 
%                         % dPhi/dO -> dPhi/dphi
%                         tmp = zeros(phi_size);
%                         tmp(row_idx:row_idx+filter_size(1)-1,:) = Om(:,:,filter_idx);
%                         dPhidphi = dPhidO(row_idx,filter_idx) * tmp;
%                     end
%                 end
% 
%                 if nn_idx ~= 1
%                     % dPhidphi -> dPhi/dO
%                     phi_dot = gradTape.("O_phi_dot"+string(nn_idx-1));
%                     dPhidO = dPhidphi .* phi_dot;
%                 end
% 
%                 % CNN updta
%                 Om_grad = Gamma * dPhidOm * error(1);
%                 B_grad = Gamma * dPhidB * error(1);
%                 % projection
%                 Om = Om + Om_grad * dt;
%                 B = B + B_grad * dt;
% 
%                 NN.("Omega"+string(nn_idx-1)) = Om;
%                 NN.("Omega_B"+string(nn_idx-1)) = B;
%             end
%         end
%     end


