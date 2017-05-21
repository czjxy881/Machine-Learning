a1 = [ones(m, 1), X];
% z2 = a1 * Theta1';
z2 = a1 * Theta1.';
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2];
% z3 = a2 * Theta2';
z3 = a2 * Theta2.';
a3 = sigmoid(z3);
% a3 remmebers the h(xi) in every row.
% the shape of a3 is 5000*10
% 将y转换为矩阵形式
Y = zeros(m, num_labels);
for i=1 : m
    Y(i, y(i)) = 1;
end

% for i =1:m
%     J = J +sum( - Y(i, :) * log(a3(i, :)') - (1 - Y(i, :)) * log(1 - a3(i, :)'));
% end
% J = J/m;
Jk = zeros(num_labels, 1);
for k=1:num_labels
    Jk(k) = (-Y(:, k).' * log(a3(:, k))) - ( (1 - Y(:, k)).' * log(1 - a3(:, k)));
end
J = sum(Jk)/m;
J = J + lambda*(sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2)))/(2*m);

% Unroll gradients

delta3 = a3 - Y;
delta2 = delta3 * Theta2.*(a2.*(1 - a2));
delta2 = delta2(:, 2: end);
Delta2 = zeros(size(delta3, 2), size(a2, 2));
Delta1 = zeros(size(delta2, 2), size(a1, 2));
for i=1:m
   Delta2 = Delta2 + delta3(i, :).' * a2(i, :);
   Delta1 = Delta1 + delta2(i, :).' * a1(i, :);
end

Theta1_grad = Delta1/m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + Theta1(:, 2:end)*(lambda/m);
Theta2_grad = Delta2/m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + Theta2(:, 2:end)*(lambda/m);
% grad = [Theta1_grad(:) ; Theta2_grad(:)];