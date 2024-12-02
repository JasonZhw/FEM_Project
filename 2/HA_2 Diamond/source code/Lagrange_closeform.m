clear all; clf;

% Original function
x = -5:5:5;
y= 1./(1+x.^2);
%x = -0.5*pi:0.01*pi:0.5*pi; 
%y= x.*sin(2*pi*x);


% Interpolated points
x_interp = linspace(x(1), x(end));

% Accurate history (for plotting and measuring errors)
y_pinterp = 1./(1+x_interp.^2);
%y_pinterp = x_interp.*sin(2*pi*x_interp); 
subplot(4,1,1)
plot(x, y, 'o');
hold on
plot(x_interp, y_pinterp, 'g-');
title('Original function');
xlabel('x');
ylabel('y');
legend("Interpolation points", "1./(1+x^2);")
%legend("Interpolationspunkte", "x.*sin(2*pi*x)")

% Lagrange interpolation (close form)
subplot(4,1,4)
y_interp = lagrange(x, y, x_interp);
plot(x, y, 'o');
hold on
plot(x_interp, y_pinterp, 'g-');
plot(x_interp, y_interp, 'x-');
plot(x_interp, y_pinterp - y_interp, 'x-');
title('Overview');
xlabel('x');
ylabel('y');
%legend('Original function', 'Interpolated function', 'Error');
hold off

% Define original function
x1 = -5:5:5;
y1 = 1./(1+x1.^2);
%x2 = -0.5*pi:0.01*pi:0.5*pi;
%y2 = x2.*sin(2*pi*x2);

% Interpolation points
x1_interp = linspace(x1(1), x1(end));
%x2_interp = linspace(x2(1), x2(end));

% Lagrange interpolation for y1 and plot
y1_interp = lagrange(x1, y1, x1_interp);
%y2_interp = lagrange(x2, y2, x2_interp);
subplot(4, 1, 2);
plot(x1, y1, 'o');
%plot(x2, y2, 'o');
hold on;
plot(x1_interp, y1_interp, 'x-');
%plot(x2_interp, y2_interp, 'x-');
title('Lagrange Interpolation for y1(x)');
%title('Lagrange Interpolation for y2(x)');
xlabel('x');
ylabel('y');
legend('Interpolation points', 'Interpolated function');
hold off;

% plot error 
subplot(4,1,3)
plot(x_interp, y_pinterp - y_interp, 'b-');
title('Error of interpolation');
xlabel('x');
ylabel('y');
legend('Error')
hold off
% Define Lagrange interpolation function
function y_interp = lagrange(x, y, x_interp)
    n = length(x);
    m = length(x_interp);
    y_interp = zeros(1, m);
    for i = 1:m
        sum = 0;
        for j = 1:n
            product = y(j);
            for k = 1:n
                if k ~= j
                    product = product .* (x_interp(i) - x(k)) ./ (x(j) - x(k));
                end
            end
            sum = sum + product;
        end
        y_interp(i) = sum;
    end
end

