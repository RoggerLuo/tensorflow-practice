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

%  拿到一个列里面 符合特定条件的元素 的index（matrix来的）
pos = find(y==1); % Return a vector of indices of nonzero elements of a matrix
neg = find(y==0);
% border color
    % k black
    % r red
% + plus
% o circle
%  markerfacecolor background color

% 用这个index matrix去X里面拿出一个新的vector,画出来，一个x轴 一个y轴
plot(X(pos,1),X(pos,2),'k+','LineWidth',4,'MarkerSize',10);
plot(X(neg,1),X(neg,2),'ro','MarkerFaceColor','r','MarkerSize',10);

% =========================================================================

hold off;

end
