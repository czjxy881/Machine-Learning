function t=try1(x,y)
% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X=[ones(size(x,1),1),x,x.^2];

% Initialize fitting parameters
initial_theta = zeros(size(X,2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = liner_costFunctionReg(initial_theta, X, y, lambda);

%fprintf('Cost at initial theta (zeros): %f\n', cost);

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and 
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(liner_costFunctionReg(t, X, y, lambda)), initial_theta, options);

[cost, grad] = liner_costFunctionReg(theta, X, y, lambda);

t=X*theta;
end


