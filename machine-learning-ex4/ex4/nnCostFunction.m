function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% h*m
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

% k*h
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% X->m*n
% Theta1->h*n
% Theta2->k*h
% y->m*1
a1=[ones(m,1) X];
z2=a1*Theta1'; % m*h
a2=sigmoid(z2);
z3=[ones(m,1) a2]*Theta2'; %m*k
a3=sigmoid(z3);

y_new=zeros(y,num_labels); %m*k
for i=1:m
	y_new(i,y(i))=1;
end;
Theta1_without_one=Theta1(:,2:end);
Theta2_without_one=Theta2(:,2:end);


% 数值运算
% J=sum(sum(-y_new.*log(a3)-(1-y_new).*log(1-a3),2))/m+ lambda/(2*m)*(sum(sum(Theta1_without_one.^2))+sum(sum(Theta2_without_one.^2)));


% 向量化，注意只有主对角线上的数据是需要的
J=sum(diag(-y_new'*log(a3)-(1-y_new)'*log(1-a3)))/m+lambda/(2*m)*(sum(diag(Theta1_without_one'*Theta1_without_one))+sum(diag(Theta2_without_one'*Theta2_without_one)));


% backpropagation
% for i=1:m
% 	a1=[1 X(i,:)];
% 	z2=a1*Theta1'; % 1*n * n*h = 1*h
% 	a2=[1 sigmoid(z2)];
% 	z3=a2*Theta2'; %1*h * h*k = 1*k 
% 	a3=sigmoid(z3);

% 	delta_3=a3-y_new(i,:); % 1*k 1*10
% 	delta_2=(delta_3*Theta2).*sigmoidGradient([1 z2]); % 1*10 * 10*25  1*h
% 	delta_2=delta_2(2:end);

% 	Theta2_grad=Theta2_grad+delta_3'*a2; % k*1 1*h  k*h
% 	Theta1_grad=Theta1_grad+delta_2'*a1; % h*1 1*n  h*n
% end


% backpropagation
a1=[ones(m,1) X]; %m*n
z2=a1*Theta1'; % m*n * n*h = m*h
a2=[ones(m,1) sigmoid(z2)];
z3=a2*Theta2'; %m*h * h*k = m*k 
a3=sigmoid(z3);
delta_3=a3-y_new; % m*k m*10
delta_2=(delta_3*Theta2).*sigmoidGradient([ones(m,1) z2]); % m*10 * 10*25  m*h
delta_2=delta_2(:,2:end); 


Theta2_grad=Theta2_grad+delta_3'*a2; % k*m m*h  k*h
Theta1_grad=Theta1_grad+delta_2'*a1; % h*m m*n  h*n










Theta1_grad=Theta1_grad/m+lambda/m*[zeros(hidden_layer_size,1) Theta1(:,2:end)];
Theta2_grad=Theta2_grad/m+lambda/m*[zeros(num_labels,1) Theta2(:,2:end)];




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
