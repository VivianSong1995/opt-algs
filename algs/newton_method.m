function [ yhist, xk ] = newton_method(x1,hess,grad,f,step,maxIter,tol,toPlot,printIters) 
%# Gradient of the logistic loss

% ------------Initialization----------------
if ~exist('maxIter','var');   maxIter=10000;      end
if ~exist('tol','var');   tol=1e-3;      end
if ~exist('printIters','var');   printIters=false;      end

% ------------Constant or Function----------------
if isa(step, 'function_handle')
    alpha = step; % f is a handle
else
    alpha = @(x) step; % f is not a handle
end

xk = x1; 
fc = f(xk);
gc = grad(xk);

xx = 1:maxIter;
yy = repmat(fc,1,maxIter);

%storage
yhist = zeros(maxIter,1);
stoppedEarly = false;

if toPlot
    clf
    h = semilogy(xx,yy,'YDataSource','yy');
end

for k = 1:maxIter
    % store
    yy(k) = f(xk);
    xk_old = xk;
    % descent direction
    d =- hess(xk)\gc;
    % step size
    ak = alpha(k);
    % update
    xk = xk + ak*d;
    gc = grad(xk);
    gnormc = norm(gc,2);
    yhist(k) = f(xk) ;
    
    if toPlot
        refreshdata(h,'caller') 
        drawnow
    end
    
    if printIters; 
        toDisp = [k, round(norm(xk-xk_old,2),4), gnormc, yy(k) ];
        fprintf('Iter %i| %06.3f | %06.3f | %08.3f \n', toDisp)
    end
    
    % stopping rule
    if gnormc < tol;    
        stoppedEarly = true;
        lastIter = k; 
        break; 
    end
end

if stoppedEarly
    yhist = yhist(1:lastIter);
end

end
