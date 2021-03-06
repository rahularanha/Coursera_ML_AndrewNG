function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

n_feats = size(X,2);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


	hypo = X * theta;
	error = hypo - y;
	%temp1 = theta(1) - alpha * ((sum((error).*X(:,1)))/m);
	%temp2 = theta(2) - alpha * ((sum((error).*X(:,2)))/m);
	%theta = [temp1;temp2];
	temp = zeros(n_feats,1);
	
	for i = 1:n_feats
		
		temp(i) = theta(i) - alpha * (sum((error).*X(:,i))/m);
	
	end
	
	theta = temp;
	

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
