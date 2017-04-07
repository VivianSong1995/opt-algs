function [ iters, xk ] = newton_method_updt(x1,hess,grad,f,maxIter,tol,toPlot,printIters,step) 
%# Gradient of the logistic loss

% ------------Initialization----------------
if ~exist('maxIter','var');   maxIter=10000;      end
if ~exist('tol','var');   tol=1e-3;      end
if ~exist('toPlot','var');   toPlot=false;      end
if ~exist('printIters','var');   printIters=false;      end
% ------------Constant or Function----------------
if ~exist('step','var');   
    btFlag=true;
else
    if isa(step, 'function_handle')
        alpha = step; % f is a handle
    else
        alpha = @(x) step; % f is not a handle
    end
end

xk=x1; 
xx=1:maxIter;
yy=repmat(f(xk),1,maxIter);

if toPlot
    clf
%    h=plot(xx,yy,'YDataSource','yy');
    h=semilogy(xx,yy,'YDataSource','yy');
end

fprintf('Iterate | DeltaX | gradNorm  | objFunc   \n')
for k =1:maxIter
    % store
    yy(k)=f(xk);
    xk_old=xk;
    gradk = grad(xk);
    % descent direction
    d=- hess(xk)\gradk;
    % step size
    if ~btFlag
        ak = alpha(k);
    else
        ak = btLineSearch(f,yy(k),xk,gradk,d);
    end
    % update
    xk = xk + ak*d;
    %
    if toPlot
        refreshdata(h,'caller') 
        drawnow
    end
    % stopping rule
    %if printIters; disp([k,round(norm(xk-xk_old,2),4), f(xk) ]); end
    if printIters
        toDisp = [k, norm(xk-xk_old,2), norm(gradk,2), yy(k) ];
        fprintf('Iter %03i| %06.3f | %09.3f | %09.3f \n', toDisp)
        %disp([k,round(norm(xc-x_old,2),4), f(xc,1:n) ]); 
    end
    if norm(xk-xk_old,2) <tol
        break
    end
end
iters = k;
end
