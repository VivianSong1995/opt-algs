function [ cost_val ] = logisticRidgeCostDivByN(w,X,y,lambda) 
%# loss function to be optimized, it's the logistic loss
[n , ~] = size(X);  
z = X*w;
yxw = y.*z;
pos = z>0;
res = zeros(n,1);
exp_z = exp( -1*z(pos) );
res(pos) = -1*yxw(pos) - log( exp_z ./ (1+exp_z)  );
res(~pos) = -1*yxw(~pos) + log( 1 + exp( z(~pos) ) );
cost_val = sum(res) / n + (lambda/2)*(w'*w) ;
end