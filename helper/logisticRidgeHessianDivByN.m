function [ Hess ] = logisticRidgeHessianDivByN(w,X,lambda) 
%Hessian of Logistic Loss
    [n,p]= size(X);
    Xw = X*w;
    z = logistic(Xw);    
    g = z .* (1 - z);
    W = sparse(1:n,1:n,g);
    XWX = X'*W*X;
    Hess = (1/n)*XWX + lambda*speye(p);
end