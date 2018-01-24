function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% 难点一 X是矩阵， 所以要循环，  
% 每一项就是Xi， 然后和theta求内积，得到原始h(x)
% 再sigmoid一下

lambda = 1;

% m,n+1  *  n+1,1 = m,1(sigmo)
sigmo  = sigmoid(X * theta); %直接使用矩阵和vector相乘   得到vector

reg = lambda / (2 * m) * (theta' * theta - theta(1)^2);
% reg = 0;

% y(m,1)
J = 1/m * ( - y' * log(sigmo) - (1 - y')* log(1-sigmo)) + reg; % 也是直接使用vector，省了循环，厉害了
% y'的意思是 y的transpose

mask = ones(size(theta));
mask(1) = 0;

% grad = ((sigmo - y)' * X)'/m;
grad = 1 / m * X' * (sigmo - y) + lambda / m * (theta .* mask);
% grad = X' * (sigmo - y)/m;

%+ lambda / m * (theta .* mask)


% regularization = lambda / m * (theta .* mask)
% thetaSize = size(theta);

%for i = 1:m
 %   hx = dot(theta,X(i,:));
  %  hx = sigmoid(hx);
  %  J = J - y(i)*log(hx)  - (1-y(i))*log(1- hx) ;
%end

%J = J/m;



% for j = 1:thetaSize(1)
%     accumulator = 0;
%     for i = 1:m
%         hx = dot(theta,X(i,:));
%         hx = sigmoid(hx);
%         accumulator = accumulator + (hx - y(i))*X(i,j) ;
%     end
%     grad(j) = accumulator/m
% end


% =============================================================

end
