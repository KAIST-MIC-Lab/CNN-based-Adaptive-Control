%% CLEANED VERSION (COMMENTED IN YOUR STYLE)
function NN = NNtrain(NN, error)

    %% ===============================================================
    %  PREPARE
    % ===============================================================
    paramCtrl  = NN.paramCtrl;                                           % PARAMETER CONTROL STRUCTURE
    gradTape   = NN.gradTape;                                            % GRADIENT TAPE FROM FORWARD PASS

    dt         = paramCtrl.dt;                                           % SIMULATION TIME STEP
    Gamma      = paramCtrl.Gamma;                                        % LEARNING RATE
    Beta       = paramCtrl.Beta;                                         % MULTIPLIER LEARNING RATE

    out_num    = paramCtrl.size_FCL_output;                              % NUMBER OF NN OUTPUTS
    Phi        = paramCtrl.NN_Out;                                       % CURRENT NN OUTPUT

    Lamda_V    = paramCtrl.Lambda_V;                                     % LAGRANGE MULTIPLIER FOR FCL
    Lamda_Om   = paramCtrl.Lambda_Om;                                    % LAGRANGE MULTIPLIER FOR CVL
    Lambda_Out  = paramCtrl.Lambda_Out;                                  % LAGRANGE FOR OUTPUT NORM

    V_norms       = paramCtrl.V_norms;                                   % FCL NORM BOUNDS
    Om_norms      = paramCtrl.Om_norms;                                  % CVL NORM BOUNDS
    NN_Out_norm   = paramCtrl.NN_Out_norm;                               % OUTPUT NORM BOUND

    FCL_num    = paramCtrl.FCL_num;                                      % NUMBER OF FCL LAYERS
    CVLon      = paramCtrl.CVLon;                                        % CVL ON/OFF FLAG

    if CVLon
        CVL_num  = paramCtrl.CVL_num;                                    % NUMBER OF CVL LAYERS
        CVL_Node = paramCtrl.CVL_Node;                                   % CVL NODE DIMENSIONS
    end


    %% ===============================================================
    %  FCL TRAINING  (FULLY CONNECTED LAYERS)
    % ================================================================
    dPhidphi = 1;                                                        % INITIAL BACKPROP SIGNAL

    for nn_idx = flip(0:FCL_num)
        V   = NN.("V"+string(nn_idx));                                   % CURRENT FCL WEIGHT MATRIX
        phi = gradTape.("FCL_phi"+string(nn_idx));                       % CURRENT LAYER INPUT
        [n,m] = size(V);                                                 % WEIGHT SHAPE

        % ---------------------------------------------------------------
        % BACKPROPAGATION ACCUMULATION
        % ---------------------------------------------------------------
        if nn_idx ~= FCL_num
            V_next  = NN.("V"+string(nn_idx+1));                         % NEXT LAYER WEIGHT
            phi_dot = gradTape.("FCL_phi_dot"+string(nn_idx+1));         % ACTIVATION DERIVATIVE
            dPhidphi = dPhidphi * (V_next' * phi_dot);                   % UPDATE CHAIN RULE
        end

        % ---------------------------------------------------------------
        % JACOBIAN W.R.T VECTORISED WEIGHT MATRIX
        % ---------------------------------------------------------------
        dPhi_dvecV = dPhidphi * kron(eye(m), phi');                      % MATRIX-TO-VECTOR JACOBIAN

        vecV = V(:);                                                     % VECTORISED WEIGHTS

        % ---------------------------------------------------------------
        % PRIMAL UPDATE (WEIGHT UPDATE)
        % ---------------------------------------------------------------
        grad_vecV = Gamma * ( ...
            dPhi_dvecV' * error ...                                      % MAIN GRADIENT TERM
        - Lamda_V(nn_idx+1) * vecV ...                                   % WEIGHT NORM CONSTRAINT TERM
        - Lambda_Out * (dPhi_dvecV' * Phi) ...                           % OUTPUT NORM CONSTRAINT TERM
        );

        vecV = vecV + grad_vecV * dt;                                    % APPLY GRADIENT UPDATE

        NN.("V"+string(nn_idx)) = reshape(vecV, n, m);                   % STORE UPDATED WEIGHTS

        % ---------------------------------------------------------------
        % DUAL UPDATE FOR FCL WEIGHT CONSTRAINT
        % ---------------------------------------------------------------
        Lamda_V(nn_idx+1) = max(0, Lamda_V(nn_idx+1) + ...
            Beta * constraint_slack(vecV, V_norms(nn_idx+1)));           % PROJECTED GRADIENT ASCENT
    end



    %% ===============================================================
    %  CVL TRAINING  (CONVOLUTIONAL LAYERS)
    % ================================================================
    if CVLon

        % ---------------------------------------------------------------
        % BACKPROP ENTERING CVL FROM FCL0
        % ---------------------------------------------------------------
        dPhi_dFCLInput = dPhidphi * NN.V0';                              % BACKPROP INTO FIRST FCL
        dPhi_dFCLInput = dPhi_dFCLInput(:,1:end-1);                      % REMOVE BIAS COLUMN

        CVLJac = struct();                                               % STRUCT TO STORE JACOBIANS

        % ===============================================================
        %  BACKPROP THROUGH CVL LAYERS
        % ===============================================================
        for idx = flip(0:CVL_num)

            phi     = gradTape.("CVL_phi"+string(idx));                  % INPUT FEATURE MAP
            if idx~= 0
                phi_dot = gradTape.("CVL_phi_dot"+string(idx));          % ACTIVATION DERIVATIVE
            else 
                phi_dot = double(1);                                             % JUNK AT INPUT LAYER
            end
                                                                 
            Om = NN.("Omega"+string(idx));                               % CVL FILTERS
            B  = NN.("Omega_B"+string(idx));                             % CVL BIASES

            filter_num = size(B,1);                                      % NUMBER OF FILTERS
            p_jc       = size(Om,1);                                     % FILTER HEIGHT
            n_jcPlus1  = CVL_Node(idx+2,1);                              % CURRENT LAYER HEIGHT
            m_jcPlus1  = CVL_Node(idx+2,2);                              % CURRENT LAYER WIDTH
            n_jc = CVL_Node(idx+1,1);                                    % PREVIOUS LAYER HEIGHT
            m_jc = CVL_Node(idx+1,2);                                    % PREVIOUS LAYER WIDTH

            % -----------------------------------------------------------
            % INITIALIZE JACOBIAN FOR LAST CVL LAYER
            % -----------------------------------------------------------
            if idx == CVL_num
                for out_idx = 1:out_num
                    CVLJac.("dPhi_"+string(out_idx)+...                 % INITIAL dPhi/dPhi_jc
                    "_dPhiC_"+string(idx)) = ...
                        reshape(dPhi_dFCLInput(out_idx,:), n_jcPlus1, m_jcPlus1); 
                end
            end

            % ===============================================================
            %  PER-OUTPUT CONVOLUTION JACOBIANS
            % ===============================================================
            for out_idx = 1:out_num

                dPhi_i_dPhi_jc = CVLJac.("dPhi_"+string(out_idx)+...    % CURRENT JACOBIAN
                "_dPhiC_"+string(idx));  

                % ----------------------------- dPhi/dOmega & dPhi/dB -----------------------------
                dPhi_i_dOm = zeros(size(Om));                           % JACOBIAN W.R.T FILTER WEIGHTS
                dPhi_i_dB  = zeros(size(B));                            % JACOBIAN W.R.T FILTER BIASES

                for k = 1:filter_num
                    Wk = Om(:,:,k);                                     % k-TH FILTER
                    dOm_k = zeros(size(Wk));                            % LOCAL GRADIENT ACCUMULATOR
                    dB_k  = 0;                                          % LOCAL BIAS GRADIENT

                    for li = 1:n_jcPlus1
                        dOm_k = dOm_k + dPhi_i_dPhi_jc(li,k) * ...
                                phi(li:li+p_jc-1,:);                    % FILTER WEIGHT GRADIENT
                        dB_k  = dB_k  + dPhi_i_dPhi_jc(li,k);           % BIAS GRADIENT
                    end

                    dPhi_i_dOm(:,:,k) = dOm_k;                          % STORE GRADIENT
                    dPhi_i_dB(k)      = dB_k;                           % STORE BIAS GRADIENT
                end

                % ----------------------------- dPhi/dphi_prev -----------------------------
                dPhi_i_dphi_jc = zeros(size(phi));                      % GRADIENT W.R.T PREVIOUS FEATURE MAP

                for lj = 1:m_jcPlus1
                    for k = 1:filter_num
                        Wk = Om(:,:,k);                                 % k-TH FILTER
                        for li = 1:n_jcPlus1
                            coef = dPhi_i_dPhi_jc(li, lj);              % LOCAL PARTIAL DERIVATIVE
                            tmp = zeros(n_jc, m_jc);                    % TEMP STORAGE
                            tmp(li:li+p_jc-1,:) = Wk;                   % PLACE FILTER INTO CORRECT REGION
                            dPhi_i_dphi_jc = dPhi_i_dphi_jc + ...
                                coef * tmp;                             % ACCUMULATE CONTRIBUTION
                        end
                    end
                end

                % STORE ALL JACOBIANS
                CVLJac.("dPhi_"+string(out_idx)+"_dOm") = dPhi_i_dOm;   % STORE FILTER WEIGHT JACOBIAN
                CVLJac.("dPhi_"+string(out_idx)+"_dB") = dPhi_i_dB;     % STORE BIAS JACOBIAN
                CVLJac.("dPhi_"+string(out_idx)+...                     % STORE INPUT GRADIENT
                "_dphiC_"+ string(idx))= dPhi_i_dphi_jc;
                % BACKPROP FOR NEXT CVL LAYER
                if idx ~= 0
                    CVLJac.("dPhi_"+string(out_idx)+...                 % APPLY ACTIVATION DERIVATIVE
                    "_dPhiC_"+ string(idx-1)) = dPhi_i_dphi_jc .* phi_dot;                               
                end
            end


            % ===============================================================
            %  ACCUMULATE GRADIENTS OVER ALL OUTPUT NODES
            % ===============================================================
            dPhi_dOm_error = zeros(size(Om));                           % ERROR CONTRIBUTION (∂Φ/∂Ω * e)
            dPhi_dB_error  = zeros(size(B));

            dPhi_dOm_Phi   = zeros(size(Om));                           % OUTPUT CONSTRAINT CONTRIBUTION (∂Φ/∂Ω * Φ)
            dPhi_dB_Phi    = zeros(size(B));

            
            for out_idx = 1:out_num                                     % ACCUM ERROR TERM
                dPhi_dOm_error = dPhi_dOm_error + ...
                    CVLJac.("dPhi_"+string(out_idx)+"_dOm") * error(out_idx); 
                dPhi_dB_error  = dPhi_dB_error + ...
                    CVLJac.("dPhi_"+string(out_idx)+"_dB")  * error(out_idx);

                dPhi_dOm_Phi   = dPhi_dOm_Phi + ...                     % ACCUM Φ TERM
                    CVLJac.("dPhi_"+string(out_idx)+"_dOm") * Phi(out_idx);    
                dPhi_dB_Phi    = dPhi_dB_Phi + ...
                    CVLJac.("dPhi_"+string(out_idx)+"_dB")  * Phi(out_idx);
            end
            % VECTORISE PARAMETERS
            theta = [Om(:); B(:)];                                      % PARAMETER VECTOR
            dtheta_error = [dPhi_dOm_error(:); dPhi_dB_error(:)];       % ERROR TERM
            dtheta_Phi   = [dPhi_dOm_Phi(:);   dPhi_dB_Phi(:)];         % OUTPUT TERM

            % ===============================================================
            %  PRIMAL UPDATE (CVL WEIGHT UPDATE)
            % ===============================================================
            grad_theta = Gamma * ( ...
                dtheta_error ...                                        % MAIN ERROR GRADIENT
            - Lamda_Om(idx+1) * theta ...                               % CVL WEIGHT NORM CONSTRAINT
            - Lambda_Out * dtheta_Phi ...                               % OUTPUT NORM CONSTRAINT
            );

            theta = theta + grad_theta * dt;                            % APPLY WEIGHT UPDATE
            % WRITE BACK TO STRUCT
            Om_size = numel(Om);
            NN.("Omega"+string(idx))   = ...
                reshape(theta(1:Om_size), size(Om));                    % UPDATE FILTERS
            NN.("Omega_B"+string(idx)) = theta(Om_size+1:end);          % UPDATE BIASES
            % ===============================================================
            %  DUAL UPDATES FOR CVL
            % ===============================================================
            Lamda_Om(idx+1) = max(0, Lamda_Om(idx+1) + ...
                Beta * constraint_slack(theta, Om_norms(idx+1)));       % CVL WEIGHT CONSTRAINT
        end
    end
    % ===============================================================
    %  DUAL UPDATE FOR OUTPUT NORM CONSTRAINT
    % ===============================================================
    Lambda_Out = max(0, Lambda_Out + ...
                Beta * constraint_slack(Phi, NN_Out_norm));             % OUTPUT NORM CONSTRAINT
    %% ===============================================================
    %  STORE UPDATED LAGRANGE MULTIPLIERS
    % ===============================================================
    NN.paramCtrl.Lambda_V = Lamda_V;
    NN.paramCtrl.Lambda_Out = Lambda_Out;
    if CVLon
        NN.paramCtrl.Lambda_Om = Lamda_Om;
    end

end

%% ===============================================================
%  CONSTRAINT FUNCTION  c(x) = (||x||² - r²)/2
% ===============================================================
function c = constraint_slack(x, radius)
    c = 0.5*(norm(x)^2 - radius^2);
end