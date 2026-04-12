% PATH PROJECTION
function ref = get_ref(Xv, Yv, track)
    % Xv, Yv: vehicle position in global frame
    R = track.R;
    L = track.L;

    if Yv < R
        if Xv >= 0 && Xv <= L
            ref = lower_straight_projection(Xv, Yv);
        elseif Xv > L
            ref = right_arc_projection(Xv, Yv, L, R);
        else
            ref = left_arc_projection(Xv, Yv, R);
        end
    else
        if Xv >= 0 && Xv <= L
            ref = upper_straight_projection(Xv, Yv, R);
        elseif Xv > L
            ref = right_arc_projection(Xv, Yv, L, R);
        else
            ref = left_arc_projection(Xv, Yv, R);
        end
    end
end

function ref = lower_straight_projection(Xv, Yv)
    ref.X_star = Xv;
    ref.Y_star = 0;
    ref.psi_des = 0;

    % lateral error
    ref.ey = Yv;
end

function ref = upper_straight_projection(Xv, Yv, R)
    ref.X_star = Xv;
    ref.Y_star = 2 * R;
    ref.psi_des = pi;

    %lateral error
    ref.ey = 2 * R - Yv;
end

function ref = right_arc_projection(Xv, Yv, L, R)
    dx = Xv - L;
    dy = Yv - R;
    theta = atan2(dy, dx);

    ref.X_star = L + R * cos(theta);
    ref.Y_star = R + R * sin(theta);
    ref.psi_des = wrap_to_pi(theta + pi / 2);

    % lateral error
    ref.ey = R - hypot(dx, dy);
end

function ref = left_arc_projection(Xv, Yv, R)
    dx = Xv;
    dy = Yv - R;
    theta = atan2(dy, dx);

    ref.X_star = R * cos(theta);
    ref.Y_star = R + R * sin(theta);

    ref.psi_des = wrap_to_pi(theta + pi / 2);

    % lateral error
    ref.ey = R - hypot(dx, dy);
end

function angle = wrap_to_pi(angle)
    angle = mod(angle + pi, 2 * pi) - pi;
end