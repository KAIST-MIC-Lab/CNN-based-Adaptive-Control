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
    FCN_act_func = paramCtrl.FCN_phi;
    CNN_act_func = paramCtrl.CNN_phi;

    CNNon = paramCtrl.CNNon;

    Dixon = paramCtrl.Dixon;
    if ~Dixon
        Ac = paramCtrl.Ac;
        rho = paramCtrl.rho;
    end

    CNN_Node = paramCtrl.CNN_Node;
    FCN_Node = paramCtrl.FCN_Node;
    
    CNN_num = paramCtrl.CNN_num;
    FCN_num = paramCtrl.FCN_num;

    FCN_weight_num = paramCtrl.FCN_weight_num;
    CNN_weight_num = paramCtrl.CNN_weight_num;

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
    fprintf("    CNN act_func: %s\n", CNN_act_func)
    fprintf("    FCN act_func: %s\n", FCN_act_func)
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

    if CNNon
        fprintf("    CNN structure\n")
for layer_idx = 1:1:CNN_num+1
        fprintf("        input %d: (%d, %d)\n", layer_idx-1, CNN_Node(layer_idx, 1), CNN_Node(layer_idx, 2))
        fprintf("                Omega %d: (%d, %d, %d)\n", ...
            layer_idx-1, CNN_Node(layer_idx, 3), CNN_Node(layer_idx, 2), CNN_Node(layer_idx, 4))
        fprintf("                Bias %d: (%d, 1)\n", ...
            layer_idx-1, CNN_Node(layer_idx, 4))
end
        fprintf("        output %d: (%d, %d)\n", layer_idx, CNN_Node(end, 1), CNN_Node(end, 2))
    end

    fprintf("    FCN structure\n")
for layer_idx = 1:1:FCN_num+1
    fprintf("        input %d: (%d, 1)\n", layer_idx-1, FCN_Node(layer_idx))
    fprintf("                V %d: (%d, %d)\n", ...
        layer_idx-1, FCN_Node(layer_idx)+1, FCN_Node(layer_idx+1))
end
    fprintf("        output %d: (%d, 1)\n", layer_idx, FCN_Node(end))

    fprintf("\n")
    
   
    % trainable variables report
    fprintf("TRAIN INFO \n")

    fprintf("    trainable variables numbers\n")
    if CNNon
        fprintf("            CNN: %d\n", CNN_weight_num)
        fprintf("            FCN: %d\n", FCN_weight_num)
        fprintf("        total: %d\n", CNN_weight_num+FCN_weight_num)
    else
        fprintf("            FCN: %d\n", FCN_weight_num)
        fprintf("        total: %d\n", FCN_weight_num)
    end
    fprintf("\n")


end