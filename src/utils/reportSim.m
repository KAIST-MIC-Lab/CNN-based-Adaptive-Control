function [] = reportSim(NN, paramSim)

    %% ===============================================================
    %  OPTIONAL LOGGING
    %% ===============================================================
    exp_name = string(paramSim.exp_name);
    if paramSim.saveResult
        result_dir = "result/" + exp_name;
        if ~exist(result_dir, 'dir')
            mkdir(result_dir);
        end
        diary(result_dir + "/" + exp_name + ".log");
    end

    %% ===============================================================
    %  PREPARE
    %% ===============================================================
    dt       = paramSim.dt;
    T        = paramSim.T;
    seed_num = paramSim.seed_num;

    paramCtrl = NN.paramCtrl;

    %% NN PARAMETERS 
    sampling_time   = paramCtrl.dt;
    stack_time      = paramCtrl.input_dt;
    learning_rate   = paramCtrl.Gamma;
    FCL_act_func    = paramCtrl.FCL_phi;
    CVL_act_func    = paramCtrl.CVL_phi;
    CVLon           = paramCtrl.CVLon;

    %% STRUCTURE AND COUNTS
    FCL_Node       = paramCtrl.FCL_Node;
    FCL_num        = paramCtrl.FCL_num; 
    FCL_weight_num = paramCtrl.FCL_weight_num;

    CVL_Node       = paramCtrl.CVL_Node;
    CVL_num        = paramCtrl.CVL_num; 
    CVL_weight_num = paramCtrl.CVL_weight_num;        

    %% TOTAL WEIGHTS
    total_weights = CVL_weight_num + FCL_weight_num;

    %% ===============================================================
    %  PRINT REPORT
    %% ===============================================================
    fprintf("===========================================\n")
    fprintf("              SIMULATION INFO              \n")
    fprintf("===========================================\n\n")

    %% SIMULATION PARAMETERS
    fprintf("SIMULATION PARAMETERS\n")
    fprintf("    Simulation Time(s): %.3f\n", T)
    fprintf("    Simulation Step Time(s): %f\n", dt)
    fprintf("    Seed Number: %d\n\n", seed_num)

    %% TRAIN PARAMETERS
    fprintf("TRAIN PARAMETERS\n")
    fprintf("    Controller Step Time(s): %f\n", sampling_time) 
    fprintf("    Input Stacking Time(s): %.3f\n", stack_time) 
    fprintf("    Learning Rate (Gamma): %.3e\n", learning_rate) 
    fprintf("    CVL Activation Function: %s\n", CVL_act_func)
    fprintf("    FCL Activation Function: %s\n\n", FCL_act_func)

    %% ===============================================================
    %  ITERATED WEIGHT NORM CONSTRAINTS  (FLEXIBLE + SAFE)
    %% ===============================================================
    fprintf("WEIGHT NORM CONSTRAINTS (L2 / FROBENIUS)\n")

    %% ---- CVL LAYERS ----
    if CVLon
        fprintf("    CONVOLUTIONAL FILTER WEIGHT NORMS:\n")

        num_Om = length(paramCtrl.Om_norms);
        max_layer_CVL = min(CVL_num + 1, num_Om);

        for idx = 1:max_layer_CVL
            fprintf("        CVL_%d  (Omega, Bias) Norm Limit = %.3e\n", ...
                idx-1, paramCtrl.Om_norms(idx));
        end

        if num_Om < CVL_num + 1
            fprintf("        (WARNING: Om_norms has only %d entries; expected %d)\n", ...
                num_Om, CVL_num + 1);
        end
        fprintf("\n")
    end

    %% ---- FCL LAYERS ----
    fprintf("    FULLY CONNECTED WEIGHT NORMS:\n")

    num_V   = length(paramCtrl.V_norms);
    max_layer_FCL = min(FCL_num + 1, num_V);

    for idx = 1:max_layer_FCL
        fprintf("        V_%d  Norm Limit = %.3e\n", ...
            idx-1, paramCtrl.V_norms(idx));
    end

    if num_V < FCL_num + 1
        fprintf("        (WARNING: V_norms has only %d entries; expected %d)\n", ...
            num_V, FCL_num + 1);
    end
    fprintf("\n")

    %% ===============================================================
    %  CVL STRUCTURE INFORMATION
    %% ===============================================================
    fprintf("CONVOLUTIONAL NEURAL NETWORK (CVL) STRUCTURE\n")
    fprintf("    Input Size: (%d, %d)\n", ...
        paramCtrl.size_CVL_input(1), paramCtrl.size_CVL_input(2))

    for layer_idx = 1:CVL_num+1
        m = CVL_Node(layer_idx, 1);
        n = CVL_Node(layer_idx, 2);
        q = CVL_Node(layer_idx, 3);
        r = CVL_Node(layer_idx, 4);

        if layer_idx <= CVL_num
            output_h = m - q + 1;
            output_w = r;
            fprintf("        LAYER %d (CONV): INPUT(%d, %d). FILTER(%d, %d). OUTPUT(%d, %d, %d)\n", ...
                layer_idx-1, m, n, q, n, output_h, output_w, r)
        else
            fprintf("        FLATTENING LAYER: INPUT SIZE → (%d, 1)\n", m*n)
        end
    end
    fprintf("\n")

    %% ===============================================================
    %  FCL STRUCTURE INFORMATION
    %% ===============================================================
    fprintf("FULLY CONNECTED LAYER (FCL) STRUCTURE\n")
    fprintf("    Input to FCL Block: (%d, 1)\n", FCL_Node(1))

    for layer_idx = 1:FCL_num+1
        n = FCL_Node(layer_idx);
        m = FCL_Node(layer_idx+1);

        if layer_idx <= FCL_num
            fprintf("        LAYER %d (V): INPUT(%d). WEIGHT MATRIX V%d SIZE = (%d, %d)\n", ...
                layer_idx-1, n, layer_idx-1, m, n+1)
        end
    end
    fprintf("    FINAL OUTPUT SIZE: (%d, 1)\n\n", FCL_Node(end))

    %% ===============================================================
    %  TRAINABLE PARAMETER COUNT
    %% ===============================================================
    fprintf("TRAINABLE PARAMETER COUNT\n")
    fprintf("    CVL PARAMETERS (Omega, Bias):  %d\n", CVL_weight_num)
    fprintf("    FCL PARAMETERS (V):            %d\n", FCL_weight_num)
    fprintf("    TOTAL PARAMETERS:              %d\n\n", total_weights)

end