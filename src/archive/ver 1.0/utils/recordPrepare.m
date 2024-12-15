
num_state = length(x);
num_input = length(u);

% pre-allocation
result.X_hist = zeros(num_state, length(t));  
result.XD_hist = zeros(num_state, length(t));  
result.U_hist = zeros(num_input, length(t));  
for Om_idx = 1:1:NN.paramCtrl.CNN_num+1
    result.Om_hist.("Om"+string(Om_idx-1)) = ...
        zeros(NN.paramCtrl.CNN_filter_size(2, Om_idx), length(t));
    result.Om_hist.("Om_B"+string(Om_idx-1)) = ...
        zeros(1, length(t));
end
result.V_hist = zeros(NN.paramCtrl.FCN_num, length(t));

% record initial values
result.X_hist(:,1) = x;
result.XD_hist(:,1) = ref_Traj(0);
result.U_hist(:,1) = u;
for Om_idx = 1:1:NN.paramCtrl.CNN_num+1
    for filter_idx = 1:1:NN.paramCtrl.CNN_Node(Om_idx, end)
        result.Om_hist.("Om"+string(Om_idx-1))(filter_idx,1) = ...
            norm(NN.("Omega"+string(Om_idx-1))(:,:,filter_idx), "fro");
    end
    result.Om_hist.("Om_B"+string(Om_idx-1))(1,1) = norm(NN.("Omega_B"+string(Om_idx-1)));
end
for V_idx = 1:1:NN.paramCtrl.FCN_num
    result.V_hist(V_idx, 1) = norm(NN.("V"+string(V_idx-1)), "fro");
end