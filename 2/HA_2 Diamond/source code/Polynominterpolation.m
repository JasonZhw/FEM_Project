clear all; clf;

% Original function
%x = -5:0.2:5; % Adjust resolution, Step: 5, 2, 1, 0.5, 0.2, 0.1
x = -0.5*pi:0.01*pi:0.5*pi;% step: 0.5π,0.2π, 0.1π, 0.05π, 0.02π, 0.01π
y = x .* sin(2*pi*x);
%y=1./(1+x.^2); %x.*sin(2*pi*x)


% Interpolated points
x_interp = linspace(x(1), x(end));

% Accurate history (for plotting and measuring errors)
%y_pinterp = 1./(1+x_interp.^2); 
y_pinterp = x_interp.*sin(2*pi*x_interp); 
subplot(4,1,1)
plot(x, y, 'o');
hold on
plot(x_interp, y_pinterp, 'g-');
legend("Interpolationspoint", " x .* sin(2*pi*x)")

% Interpolation with built-in method
% (uses least squares if necessary)
tic
p = polyfit(x, y, size(x,2));
y_interp = polyval(p, x_interp);
toc

subplot(4,1,2)
plot(x, y, 'o');
hold on
plot(x_interp, y_interp, 'x-');
legend("Interpolationspoint","Interpolated function")
subplot(4,1,3)
plot(x, y, 'o');
hold on
plot(x_interp, y_pinterp - y_interp, 'y-');
legend("Interpolationspoint",'Error')

subplot(4,1,4)
plot(x, y, 'o');
hold on
plot(x_interp, y_pinterp, 'g-');
plot(x_interp, y_interp, 'x-');
plot(x_interp, y_pinterp - y_interp, 'y-');
legend('Interpolationspoint - Least squares', ' x .* sin(2*pi*x)', 'Interpolated function', 'Error')