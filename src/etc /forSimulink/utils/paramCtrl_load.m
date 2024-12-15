function NN = paramCtrl_load(paramSim)
    %% Check! Check! Critical Setting
    % NN.paramCtrl.CNNon = 0;
    NN.paramCtrl.CNNon = 1;
    % NN.paramCtrl.RobustOn = 0;
    NN.paramCtrl.RobustOn = 1;

    NN.paramCtrl.Dixon = 0;
    % NN.paramCtrl.Dixon = 1;

    NN.paramCtrl.Gamma = 1e+3; % learning rate
    % NN.paramCtrl.Gamma = 1e+1; % learning rate

    %% DIXON; no Ac, rho
    if NN.paramCtrl.Dixon
        NN.paramCtrl.rho = 0; % e-modification
        NN.paramCtrl.Ac = -eye(2);
    else
        NN.paramCtrl.rho = 10+0; % e-modification
        % NN.paramCtrl.rho = 50+0; % e-modification
        NN.paramCtrl.Ac = diag([-10,-10]);
        % NN.paramCtrl.Ac = diag([-50,-50]);
    end

    %% Controller's PARAMETERS
    % sampling time for controller
    NN.paramCtrl.dt = paramSim.dt;
    NN.paramCtrl.input_dt = 1e-1;
    % NN.paramCtrl.input_dt = 1e-2;

    % for global stability
    if NN.paramCtrl.RobustOn
        NN.paramCtrl.ks = 1;
    else
        NN.paramCtrl.ks = 0;
    end

    NN.paramCtrl.inv_Ac = inv(NN.paramCtrl.Ac);

    %% NN SIZE
    NN.paramCtrl.size_CNN_input = [10, 6]; 
    % NN.paramCtrl.size_FCN_input = 6; % if CNN on, auto-determined
    NN.paramCtrl.size_FCN_input = NN.paramCtrl.size_CNN_input(end); % if CNN on, auto-determined
    NN.paramCtrl.size_FCN_output = 2; 
            
    %% PARAMETERS AND OTHER SETTING
    NN.paramCtrl.FCN_phi = "tanh";
    % NN.paramCtrl.FCN_phi = "relu";

    NN.paramCtrl.CNN_phi = "tanh";
    % NN.paramCtrl.CNN_phi = "relu";

    NN.paramCtrl.FCN_radius = 100;
    NN.paramCtrl.CNN_radius = 100;

    %% CNN STRUCUTRE
    NN.paramCtrl.CNN_filter_size = ...
        [
        % q(filter height),     r(filter number)
%         5       3
        5       2
        3       2
        ];
    
    NN.paramCtrl.CNN_Node = zeros(size(NN.paramCtrl.CNN_filter_size, 1)+1, 4);
    % m(input height,    n(input width),    q(filter height),    r(filter number)
    % - input layer
    % - hidden layers
    % - output layer (q, r are is trash values; 0)
    NN.paramCtrl.CNN_Node(1:end-1, 3:4) = NN.paramCtrl.CNN_filter_size;
    NN.paramCtrl.CNN_Node(1,1:2) = NN.paramCtrl.size_CNN_input;
    NN.paramCtrl.CNN_Node(2:end,2) = NN.paramCtrl.CNN_filter_size(1:end, 2);
    for CNN_idx = 2:1:size(NN.paramCtrl.CNN_filter_size, 1)+1
        NN.paramCtrl.CNN_Node(CNN_idx, 1) = ...
            NN.paramCtrl.CNN_Node(CNN_idx-1, 1) - NN.paramCtrl.CNN_Node(CNN_idx-1, 3) + 1;
    end

    %% FCN STRUCUTRE
    if NN.paramCtrl.CNNon
        NN.paramCtrl.FCN_Node = ...
            [
            % concanate matrix output from CNN
            NN.paramCtrl.CNN_Node(end,1)*NN.paramCtrl.CNN_Node(end,2)
    %         NN.paramCtrl.size_FCN_input
            8
            8
            NN.paramCtrl.size_FCN_output
            ];  
    else
        NN.paramCtrl.FCN_Node = ...
            [
            NN.paramCtrl.size_FCN_input
            8
            8
            8
            4
            NN.paramCtrl.size_FCN_output
            ];  
    end

    %% PASSIVE PARAMETERS
    % ..._num means k_f, k_c index numbers
    % subtract 2 to exclude input/output layer
    NN.paramCtrl.FCN_num = length(NN.paramCtrl.FCN_Node)-2;
    NN.paramCtrl.CNN_num = size(NN.paramCtrl.CNN_Node, 1)-2; 

    NN.paramCtrl.FCN_weight_num = 0;
    NN.paramCtrl.CNN_weight_num = 0;

    %% DEBUG
    % checkNNsize(NN.paramCtrl);


end

%% LOCAL FUNCTIONS
function [] = checkNNsize(paramCtrl)
    % error("Check the size of CNN's output and input of FCN (must have same value)")
end