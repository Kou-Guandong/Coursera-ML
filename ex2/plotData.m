function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


positiveIndex = find(y==1);
negativeIndex = find(y==0);


plot( X(positiveIndex, 1), X(positiveIndex, 2), 'g+', 'LineWidth', 2, ...
    'MarkerSize', 6);
plot( X(negativeIndex, 1), X(negativeIndex, 2), 'ro', 'MarkerFaceColor', 'r', ...
    'MarkerSize', 6);



% =========================================================================



hold off;

end
