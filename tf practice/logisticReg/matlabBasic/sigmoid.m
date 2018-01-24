function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


%  size函数的意思是获取这个矩阵的row和column的数量
%  现循环row，再在current row里面循环column， 达到遍历每一个的目的 //话说，反过来也可以啊..
% 然后计算 赋值
% 最后记得加 semicolon  //by adding a semicolon at the end of your statement it will suppress the intermediate result

theSize  = size(z);
theRow = theSize(1);
theColumn = theSize(2);

for i = 1:theRow
    for j = 1:theColumn
        g(i,j) =  1/( 1 + e.^(  -z(i,j) ));
    end
end



% =============================================================

end
