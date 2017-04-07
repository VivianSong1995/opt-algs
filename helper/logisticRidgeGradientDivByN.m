function [ grad ] = logisticRidgeGradientDivByN(w,X,y,lambda) 
    %# Gradient of the logistic loss
    [n , ~] = size(X);  
    Xw = X*w;
    h = logistic( Xw );
    diff = h - y;
    grad = (1/n)*(X'*diff) +lambda*w;
end
