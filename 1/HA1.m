%% 
x = 1 + 2 + 3 + 4*2    % 计算表达式并赋值给变量 x（结果为 14）
y = 1:5                 % 创建一个从 1 到 5 的向量（结果为 [1, 2, 3, 4, 5])
z = x + y;              % 将变量 x 和向量 y 相加（结果为 [15, 16, 17, 18, 19])
z2 = x + y              % 重复上一行代码，没有效果

x = 1:10                % 创建一个从 1 到 10 的向量（结果为 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = x.^2                % 对向量 x 中的每个元素求平方（结果为 [1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
plot(x, y)              % 绘制 x 和 y 的散点图
plot(x, y, 'go')        % 绘制 x 和 y 的散点图，并使用绿色圆圈标记数据点
plot(x, y, 'rx-')       % 绘制 x 和 y 的折线图，并使用红色实心 x 标记数据点

%% 
s = [1 2 3] 
t = [1; 2; 3]

A = [1 2 3; 4 5 6; 7 8 9]
A*t

A.*s
s'
A'
A*s'
t*s
s*t
s'.* t 
A^2 
A.^2