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


J = 1/m * sum(getErrorArray(theta, X, y));


grad =  getGrad(theta, X, y);

% =============================================================

end

function errArray = getErrorArray(theta, X, y)

errArray = -y' * log(sigmoid(X * theta)) - (1-y)' * log(1 - sigmoid(X * theta));

end

function grad = getGrad(theta, X, y)

grad = 1/length(y) * sum( X .* repmat(( sigmoid(X*theta) - y), 1, size(X,2) ) );

end