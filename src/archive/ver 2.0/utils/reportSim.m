function [] = reportSim(NN, paramSim)
    %% LOGGING
    exp_name = string(paramSim.exp_name);
    
    if paramSim.saveResult
        result_dir = "result/"+exp_name;
        if ~exist(result_dir, 'dir')
           mkdir(result_dir);
        end
    
       diary (result_dir+"/"+exp_name+".log");
    end
    %% PREPARE
    dt = paramSim.dt;
    T = paramSim.T;
    seed_num = paramSim.seed_num;

    paramCtrl = NN.paramCtrl;

    robust_term = paramCtrl.ks;
    sampling_time = paramCtrl.dt;
    stack_time = paramCtrl.input_dt;

    learning_rate = paramCtrl.Gamma;
    FCL_act_func = paramCtrl.FCL_phi;
    CVL_act_func = paramCtrl.CVL_phi;

    CVLon = paramCtrl.CVLon;
    LSTMon = paramCtrl.LSTMon;

    Dixon = paramCtrl.Dixon;
    if ~Dixon
        Ac = paramCtrl.Ac;
        rho = paramCtrl.rho;
    end
    
    if CVLon
        CVL_Node = paramCtrl.CVL_Node;
        CVL_num = paramCtrl.CVL_num;        
    end

    if LSTMon
        bc = paramCtrl.bc;
        bh = paramCtrl.bh;
        h0 = paramCtrl.LSTM_in_size;
        h1 = paramCtrl.LSTM_CS_size;
        h2 = paramCtrl.LSTM_N_size;
    end

    FCL_Node = paramCtrl.FCL_Node;
    FCL_num = paramCtrl.FCL_num;

    FCL_weight_num = paramCtrl.FCL_weight_num;
    CVL_weight_num = paramCtrl.CVL_weight_num;
    LSTM_weight_num = NN.paramCtrl.LSTM_weight_num;


    %% MAIN REPORT
    fprintf("===========================================\n")
    fprintf("              SIMULATION INFO              \n")
    fprintf("===========================================\n")
    fprintf("\n")
    
    % simulation parameters
    fprintf("SIMULATION PARAMETERS \n")
   
    fprintf("    Simulation Time(s): %.3f\n", T)
    fprintf("    Simulation Step Time(s): %f\n", dt)
    fprintf("    Seed Number: %d\n", seed_num)
    fprintf("\n")

    % train parameters
    fprintf("TRAIN PARAMETERS \n")
   
    fprintf("    Sampling Time(s): %f\n", sampling_time)
    fprintf("    Stracking Time(s): %.3f\n", stack_time)
    fprintf("\n")
    fprintf("    Learning Rate: %.3f\n", learning_rate)
    fprintf("    CVL act_func: %s\n", CVL_act_func)
    fprintf("    FCL act_func: %s\n", FCL_act_func)
    fprintf("\n")
    fprintf("    robust term: %.2f\n", robust_term)
    fprintf("\n")

    if ~Dixon
        fprintf("    From E.K.\n")
        fprintf("        Damping Factor: %f\n", rho)
        fprintf("        Hurwitz Mat: \n")
        disp(Ac)
    else
        fprintf("    From Dixon\n")
        fprintf("        (No e-modification and Hurwitz Matrix)\n")
    end


    % structure report
    fprintf("STRUCTURE REPORT \n")

    if CVLon
        fprintf("    CVL structure\n")
for layer_idx = 1:1:CVL_num+1
        fprintf("        input %d: (%d, %d)\n", layer_idx-1, CVL_Node(layer_idx, 1), CVL_Node(layer_idx, 2))
        fprintf("                Omega %d: (%d, %d, %d)\n", ...
            layer_idx-1, CVL_Node(layer_idx, 3), CVL_Node(layer_idx, 2), CVL_Node(layer_idx, 4))
        fprintf("                Bias %d: (%d, 1)\n", ...
            layer_idx-1, CVL_Node(layer_idx, 4))
end
        fprintf("        output %d: (%d, %d)\n", layer_idx, CVL_Node(end, 1), CVL_Node(end, 2))
    end

    if LSTMon
        fprintf("   LSTM structure\n")
        fprintf("        Cell state gain %f\n", bc)
        fprintf("        Hidden state gain %f\n", bh)
        fprintf("\n")
        fprintf("        input: (%d, 1)\n", h0)
        fprintf("        cell, hidden state: (%d, 1)\n", h1)
    end

    fprintf("    FCL structure\n")
for layer_idx = 1:1:FCL_num+1
    fprintf("        input %d: (%d, 1)\n", layer_idx-1, FCL_Node(layer_idx))
    fprintf("                V %d: (%d, %d)\n", ...
        layer_idx-1, FCL_Node(layer_idx)+1, FCL_Node(layer_idx+1))
end
    fprintf("        output %d: (%d, 1)\n", layer_idx, FCL_Node(end))

    fprintf("\n")
    
   
    % trainable variables report
    fprintf("TRAIN INFO \n")

    fprintf("    trainable variables numbers\n")
    if CVLon
        fprintf("            CVL: %d\n", CVL_weight_num)
    end
    if LSTMon
        fprintf("            LSTM: %d\n", LSTM_weight_num)
    end
        fprintf("            FCL: %d\n", FCL_weight_num)
        fprintf("        total: %d\n", CVL_weight_num+LSTM_weight_num+FCL_weight_num)
    fprintf("\n")


end