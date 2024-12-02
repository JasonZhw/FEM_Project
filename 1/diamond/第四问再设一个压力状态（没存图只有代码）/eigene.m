%% 
% Bereich der x- und y-Werte definieren
[x, y] = meshgrid(-10:0.1:10, -10:0.1:10);
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

% 计算散度的各个部分
% 首先对 sigma_xx 沿着 y 方向（第二维）差分，对 sigma_xy 沿着 x 方向（第一维）差分
temp1 = diff(sigma_xx, 1, 2) / dx; % 结果尺寸为 201x200
temp2 = diff(sigma_xy, 1, 1) / dx; % 结果尺寸为 200x201
% 调整尺寸以使尺寸匹配
temp1 = temp1(1:end-1, :); % 从 201x200 裁剪到 200x200 不要最后一行
temp2 = temp2(:, 1:end-1); % 从 200x201 裁剪到 200x200 不要最后一列
sz1=size(temp1)
sz2=size(temp2)
% 计算 div_sigma_x_num
div_sigma_x_num = temp1 + temp2; % 这里两个矩阵的尺寸都是 200*200
% 同样地处理 div_sigma_y_num
temp3 = diff(sigma_xy, 1, 2) / dx;
temp4 = diff(sigma_yy, 1, 1) / dx; 
% 调整尺寸以使尺寸匹配
temp3 = temp3(1:end-1, :); 
temp4 = temp4(:, 1:end-1); 
% 计算 div_sigma_y_num
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
% 定义应力张量的分量
sigma_xx = x.^2 - 5*x.*y - 7*y.^2;
sigma_yy = 3*x.^2 + x.*y + 5*y.^2;
sigma_xy = 4*x.^2 - 3*x.*y - 2*y.^2;
% 计算散度
div_sigma_x = diff(sigma_xx, x) + diff(sigma_xy, y);
div_sigma_y = diff(sigma_xy, x) + diff(sigma_yy, y);

% 创建数值网格
[x_num, y_num] = meshgrid(-10:0.1:10, -10:0.1:10); 

% 在数值网格上评估散度的 x 和 y 分量
div_sigma_x_sym = double(subs(div_sigma_x, {x, y}, {x_num, y_num}));
div_sigma_y_sym = double(subs(div_sigma_y, {x, y}, {x_num, y_num}));

% 显示结果
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

% 显示误差的均值
fprintf('Der durchschnittliche Fehler der x-Komponente der Divergenz σ: %.4f\n', mean(err_x(:)));

fprintf('Der durchschnittliche Fehler der y-Komponente der Divergenz σ: %.4f\n', mean(err_y(:)));

% 绘制数值散度和符号散度的差异图
figure;
subplot(1,2,1);
imagesc(err_x);
colorbar;
title('Der Unterschied zwischen numerischer und symbolischer Divergenz (die x-Komponente von σ)');

subplot(1,2,2);
imagesc(err_y);
colorbar;
title('Der Unterschied zwischen numerischer und symbolischer Divergenz (die y-Komponente von σ)');






