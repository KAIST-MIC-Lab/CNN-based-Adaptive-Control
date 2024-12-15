function Y = CNN1D(X, Omega, Bias)
    % X is matrix
    % Omega is set of filter or matrix or vector 
    % B is bias of Omega

    m = size(X); m = m(1);
    [q, n, r] = size(Omega); % for now, Omega is set of matrix (i.e. 3D)

    %% DEBUG
    if isvector(X)
        error("1D-CNN's input(X) is vector")
    end

    if isvector(Omega)
        % Omega is matrix and 
        error("1D-CNN's filter(Omega) is vector")
    end

    %% CALC
    size_Y = [m-q+1, r];
    Y = zeros(size_Y);

    for filter_idx = 1:1:r
        filter = Omega(:,:,filter_idx);
        b = Bias(filter_idx);

        for row_idx = 1:1:size_Y(1)
            Y(row_idx, filter_idx) = ...
                sum(X(row_idx: row_idx+q-1, :) .* filter, "all") + b;    
        end
    end   
end