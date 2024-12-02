%% 
% Bereich der x- und y-Werte definieren
[x, y] = meshgrid(-10:0.1:10, -10:0.1:10);
dx = 0.1;
% Spannungen berechnen
sigma_xx = x.^2 - 5*x.*y - 7*y.^2;
sigma_yy = 3*x.^2 + x.*y + 5*y.^2;
sigma_xy = 4*x.^2 - 3*x.*y - 2*y.^2;
% von-Mises-Vergleichsspannung berechnen
sigma_v = sqrt(0.5*((sigma_xx - sigma_yy).^2) + 3*(sigma_xy.^2));
% Plots erstellen
figure;
subplot(2,2,1);
scatter(x(:), y(:), 10, sigma_xx(:), 'filled');
title('\sigma_{xx}');
colorbar;
subplot(2,2,2);
scatter(x(:), y(:), 10, sigma_yy(:), 'filled');
title('\sigma_{yy}');
colorbar;
subplot(2,2,3);
scatter(x(:), y(:), 10, sigma_xy(:), 'filled');
title('\sigma_{xy}');
colorbar;
subplot(2,2,4);
scatter(x(:), y(:), 10, sigma_v(:), 'filled');
title('von Mises');
colorbar;
%% 
temp1 = diff(sigma_xx, 1, 2) / dx; 
temp2 = diff(sigma_xy, 1, 1) / dx; 
temp1 = temp1(1:end-1, :); 
temp2 = temp2(:, 1:end-1); 
sz1=size(temp1)
sz2=size(temp2)

div_sigma_x_num = temp1 + temp2; 

temp3 = diff(sigma_xy, 1, 2) / dx;
temp4 = diff(sigma_yy, 1, 1) / dx; 

temp3 = temp3(1:end-1, :); 
temp4 = temp4(:, 1:end-1); 

div_sigma_y_num = temp3 + temp4; 
figure;
subplot(1,2,1);
imagesc(div_sigma_x_num);
colorbar;
title('x-Komponente der numerischen Divergenz σ');
subplot(1,2,2);
imagesc(div_sigma_y_num);
colorbar;
title('y-Komponente der numerischen Divergenz σ');
%% 

syms x y
sigma_xx = x.^2 - 5*x.*y - 7*y.^2;
sigma_yy = 3*x.^2 + x.*y + 5*y.^2;
sigma_xy = 4*x.^2 - 3*x.*y - 2*y.^2;

div_sigma_x = diff(sigma_xx, x) + diff(sigma_xy, y);
div_sigma_y = diff(sigma_xy, x) + diff(sigma_yy, y);


[x_num, y_num] = meshgrid(-10:0.1:10, -10:0.1:10); 

div_sigma_x_sym = double(subs(div_sigma_x, {x, y}, {x_num, y_num}));
div_sigma_y_sym = double(subs(div_sigma_y, {x, y}, {x_num, y_num}));


disp('x-Komponente der Divergenz σ:');
pretty(div_sigma_x)
disp('y-Komponente der Divergenz σ:');
pretty(div_sigma_y)

figure;
subplot(1,2,1);
imagesc(div_sigma_x_sym);
colorbar;
title('x-Komponente der symbolischen Divergenz σ');

subplot(1,2,2);
imagesc(div_sigma_y_sym);
colorbar;
title('y-Komponente der symbolischen Divergenz σ');

%% 
err_x = abs(div_sigma_x_num - div_sigma_x_sym(1:end-1,1:end-1));
err_y = abs(div_sigma_y_num - div_sigma_y_sym(1:end-1,1:end-1));


fprintf('Der durchschnittliche Fehler der x-Komponente der Divergenz σ: %.4f\n', mean(err_x(:)));

fprintf('Der durchschnittliche Fehler der y-Komponente der Divergenz σ: %.4f\n', mean(err_y(:)));

figure;
subplot(1,2,1);
imagesc(err_x);
colorbar;
title('Der Unterschied zwischen numerischer und symbolischer Divergenz (die x-Komponente von σ)');

subplot(1,2,2);
imagesc(err_y);
colorbar;
title('Der Unterschied zwischen numerischer und symbolischer Divergenz (die y-Komponente von σ)');






