function x_next = RK4th(sys, input, x, t, dt)
    u = input(x, t);
    k1 = reshape(sys(x, t, u), size(x));
    k2 = reshape(sys(x + k1*dt/2, t + dt/2, u), size(x));
    k3 = reshape(sys(x + k2*dt/2, t + dt/2, u), size(x));
    k4 = reshape(sys(x + k3*dt, t + dt, u), size(x));

    x_next = x + (k1 + 2*k2 + 2*k3 + k4) / 6 * dt;
end