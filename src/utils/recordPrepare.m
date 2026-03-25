num_state = length(x);
num_input = length(u);
num_outs= length(y);

%% ============================================================
%    PREALLOCATE RECORDING MATRICES
%% ============================================================
result.Y_hist  = zeros(num_outs, length(t));
result.YD_hist = zeros(num_outs, length(t));  
result.U_hist  = zeros(num_input, length(t));  
result.E_hist  = zeros(num_outs, length(t));
result.XY_pos= zeros(2, length(t));                                                 % FOR RECORDING XY POSITION OF THE VEHICLE                             
result.XY_pos_des = zeros(2, length(t));                                             % FOR RECORDING DESIRED XY POSITION OF THE VEHICLE

%% ====== CVL WEIGHT HISTORY CONTAINER (STRUCT OF ARRAYS) ======
result.Om_hist = struct();
if NN.paramCtrl.CVLon
    for Om_idx = 1 : NN.paramCtrl.CVL_num+1

        % NUMBER OF FILTERS IN THIS LAYER
        num_filters = NN.paramCtrl.CVL_filter_size(Om_idx, 2);

        % STORE FILTER NORM HISTORY
        result.Om_hist.("Om"+string(Om_idx-1)) = ...
            zeros(num_filters, length(t));

        % STORE BIAS NORM HISTORY
        result.Om_hist.("Om_B"+string(Om_idx-1)) = ...
            zeros(1, length(t));
    end
end

%% ====== FCL WEIGHT NORM HISTORY ======
result.V_hist = zeros(NN.paramCtrl.FCL_num+1, length(t));

%% ============================================================
%    INITIAL RECORDING (k = 1)
%% ============================================================
result.Y_hist(:,1)  = y;
result.YD_hist(:,1) = yd;
result.U_hist(:,1)  = u;

%% ====== RECORD CVL WEIGHT NORMS AT TIME INDEX 1 ======
if NN.paramCtrl.CVLon
    for Om_idx = 1 : NN.paramCtrl.CVL_num+1

        % GET WEIGHT MATRIX (3D): HEIGHT x WIDTH x NUM_FILTERS
        Om = NN.("Omega"+string(Om_idx-1));

        % NUMBER OF FILTERS
        num_filters = size(Om, 3);

        % RECORD FROBENIUS NORM FOR EACH FILTER
        for filter_idx = 1 : num_filters
            result.Om_hist.("Om"+string(Om_idx-1))(filter_idx, 1) = ...
                norm(Om(:,:,filter_idx), "fro");                     % RECORD FILTER F-NORM
        end

        % RECORD BIAS NORM
        B = NN.("Omega_B"+string(Om_idx-1));
        result.Om_hist.("Om_B"+string(Om_idx-1))(1, 1) = norm(B);
    end
end

%% ====== RECORD COMBINED NORM OF (Om + B) ======
if NN.paramCtrl.CVLon
    for Om_idx = 1 : NN.paramCtrl.CVL_num+1

        Om = NN.("Omega"+string(Om_idx-1));
        B  = NN.("Omega_B"+string(Om_idx-1));

        % FLATTEN WEIGHTS AND BIAS
        Om_flat = Om(:);
        B_flat  = B(:);

        Combined = [Om_flat; B_flat];

        % STORE AS: Om_Combined0, Om_Combined1, ...
        result.Om_hist.("Om_Combined"+string(Om_idx-1)) = ...
            zeros(1, length(t));      % PREALLOCATE ROW VECTOR

        result.Om_hist.("Om_Combined"+string(Om_idx-1))(1) = ...
            norm(Combined, "fro");    % RECORD INITIAL COMBINED NORM
    end
end

%% ====== RECORD FCL WEIGHTS AT TIME 1 ======
for V_idx = 1 : NN.paramCtrl.FCL_num+1
    result.V_hist(V_idx, 1) = norm(NN.("V"+string(V_idx-1)), "fro");
end
