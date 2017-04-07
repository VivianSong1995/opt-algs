function alpha = btLineSearch(objFunc, objFuncValue, x, grad, dir)
% Backtracking line search using armijo criterion
% objFunValue so no need to double eval
%
% objFunc      - handle for objective function
% objFuncValue - current objective function value @ x
% x            - current x
% dx           - dx
% dir          - search direction
%
% example : btLineSearch(objFunc,objFuncValue,x,dx,dir)

alphaBar     = 1; % this is the maximum step length
alpha        = alphaBar;
fac          = 1/2; % < 1 reduction factor of alpha
c_1          = 1e-4;

while objFunc(x + alpha*dir) > objFuncValue + (c_1*alpha)*(dir'*grad);
    
    alpha = fac*alpha;
    
    if alpha < 10*eps
        error('Error in Line search - alpha close to working precision');
    end
    
end

end