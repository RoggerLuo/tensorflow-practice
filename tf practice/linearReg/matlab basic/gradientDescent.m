function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% roger
% 计算每次的导数，然后update theta1 theta2
% for iter = 1:num_iters
%     for i = columns(x):
%         thetaAccu = 0
%         for j = m:
%             thetaAccu = thetaAccu + (dot(X(j,:),theta) - y)*X[i]
%         end
%         theta(i) = theta(i) - thetaAccu*alpha
%     end
% end

% mine is even better
% roger

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    sum1 = 0;
    for i = 1:m
        sum1 = sum1 + (dot(X(i,:),theta) - y(i))*X(i,[1]);
    end
    sum1 = sum1*alpha/m;

    sum2 = 0;
    for i = 1:m
        sum2 = sum2 + (dot(X(i,:),theta) - y(i))*X(i,[2]);
    end
    
    sum2 = sum2*alpha/m;


    theta(1) = theta(1) - sum1;
    theta(2) = theta(2) - sum2;
    theta;
    if mod(iter,200) == 0
        hold on; % keep previous plot visible
        plot(X(:,2), X*theta, '-');
        hold off % don't overlay any more plots on this figure
        current_J = computeCost(X, y, theta)
    end


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
