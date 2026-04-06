function Ref = get_ref(t, Plant)
    %% PREPARE
    vx = Plant.vx;
    dist = vx * t;                                                % TOTAL DISTANCE TRAVELED ALONG THE PATH
    
    %% USER PARAMETERS
    R1 = 10;                                                      % RADIUS OF THE UPPER SEMICIRCLES
    R2 = 10;                                                      % RADIUS OF THE LOWER SEMICIRCLES
    L  = 5;                                                       % LENGTH OF THE STRAIGHT SECTIONS
     
    %% SEGMENT BOUNDARIES CALCULATION
    S1 = L;                                                       % END OF 1ST STRAIGHT
    S2 = S1 + pi*R1;                                              % END OF 1ST SEMICIRCLE (UP)
    S3 = S2 + L;                                                  % END OF 2ND STRAIGHT
    S4 = S3 + pi*R2;                                              % END OF 2ND SEMICIRCLE (DOWN)

    %% PIECEWISE REFERENCE CALCULATION
    if dist < S1
        % Segment 1: Straight
        Ref.psi_dot_des  = 0;                                     
        Ref.psi_ddot_des = 0;
        Ref.psi_des      = 0;                                     % HEADING REMAINS 0
        Ref.X_des        = dist;                                  % X INCREASES LINEARLY
        Ref.Y_des        = 0;                                     % Y REMAINS 0
        
    elseif dist < S2
        % Segment 2: Semicircle UP 
        theta            = (dist - S1) / R1;                      % LOCAL ANGLE ALONG ARC (0 to PI)
        Ref.psi_dot_des  = vx / R1;                               
        Ref.psi_ddot_des = 0;
        Ref.psi_des      = theta;                                 % HEADING ROTATES FROM 0 TO PI
        Ref.X_des        = S1 + R1 * (1 - cos(theta));            % X DURING VERTICAL ARC
        Ref.Y_des        = R1 * sin(theta);                       % Y DURING VERTICAL ARC
        
    elseif dist < S3
        % Segment 3: Straight (Returning at heading PI)
        Ref.psi_dot_des  = 0;
        Ref.psi_ddot_des = 0;
        Ref.psi_des      = pi;                                    % HEADING IS NOW CONSTANT PI (BACKWARDS)
        Ref.X_des        = S1 - (dist - S2);                      % X DECREASES
        Ref.Y_des        = 2 * R1;                                % Y IS AT TOP LANE
        
    elseif dist < S4
        % Segment 4: Semicircle DOWN 
        theta            = (S4-dist) / R2;                      % LOCAL ANGLE (0 to PI)
        Ref.psi_dot_des  = -vx / R2;                              
        Ref.psi_ddot_des = 0;
        Ref.psi_des      = pi - theta;                            % HEADING ROTATES PI TO 0
        Ref.X_des        = S1 + R1+ R2(1-cos(theta));                % X DURING VERTICAL ARC
        Ref.Y_des        = -R2 * sin(theta);                       % Y DECREASES BACK TO 0

    else
        % Back to Straight
        Ref.psi_dot_des  = 0;
        Ref.psi_ddot_des = 0;
        Ref.psi_des      = 0;
        Ref.X_des        = (S1 - L) - (dist - S4);                % CONTINUING STRAIGHT
        Ref.Y_des        = 0;
    end
end