clear all; clf;

% Original function
%x = -5:0.1:5; % Adjust resolution, Step: 5, 2, 1, 0.5, 0.2, 0.1
%y = 1./(1+x.^2);
x = -0.5*pi:0.05*pi:0.5*pi;% step: 0.5π,0.2π, 0.1π, 0.05π, 0.02π, 0.01π
y = x .* sin(2*pi*x);

% Interpolated points
x_interp = linspace(x(1), x(end));

% Accurate history (for plotting and measuring errors)
%y_pinterp = 1./(1+x_interp.^2); 
y_pinterp = x_interp.*sin(2*pi*x_interp);
subplot(4,1,1)
plot(x, y, 'o');
hold on
plot(x_interp, y_pinterp, 'g-');
%legend("Interpolation points", "1./(1+x^2);")
legend("Interpolation points", "sin(2*pi*x);")

% Lagrangian interpolation (piecewise, with square polynomials)
y_interp = lagrange_stepwise(x, y, x_interp);

subplot(4,1,2)
plot(x, y, 'o');
hold on
plot(x_interp, y_interp, 'x-');
legend("Interpolation points","Interpolated function")

subplot(4,1,3)
hold on
plot(x_interp, y_pinterp - y_interp, 'b-');
legend('Error')

subplot(4,1,4)
plot(x, y, 'o');
hold on
plot(x_interp, y_pinterp, 'g-');
plot(x_interp, y_interp, 'x-');
plot(x_interp, y_pinterp - y_interp, 'b-');
%legend("Interpolation points", "1./(1+x^2)", "Lagrange interpolation","Error")
legend("Interpolation points", "x .* sin(2*pi*x)", "Lagrange interpolation")


% Function for piecewise Lagrangian interpolation with quadratic polynomials
function y_interp = lagrange_stepwise(x, y, x_interp)
    n = length(x);
    y_interp = zeros(size(x_interp));
    
    for i = 1:n-1
        xi = x(i:i+1); % Get current point and next point
        yi = y(i:i+1); % The corresponding function value
        for j = 1:length(x_interp)
            if x_interp(j) >= xi(1) && x_interp(j) <= xi(end)
                y_interp(j) = lagrange_poly(xi, yi, x_interp(j));
            end
        end
    end
end

function yi = lagrange_poly(x, y, xi)
    n = length(x);
    yi = 0;
    for i = 1:n
        L = 1;
        for j = 1:n
            if i ~= j
                L = L * (xi - x(j)) / (x(i) - x(j));
            end
        end
        yi = yi + y(i) * L;
    end
end
