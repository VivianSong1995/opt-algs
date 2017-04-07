function [ yhist, xc ] = gradientDescent(x0,f,g,step,maxIter,tol,printIters) 
%# Gradient of the logistic loss

% ------------Initialization----------------
if ~exist('maxIter','var');     maxIter = 1e6;      end
if ~exist('tol','var');         tol = 1e-4;           end
if ~exist('printIters','var');  printIters = false;   end

% initialization
xc = x0;
d  = -g(xc);
%ak = step;  % step size

%storage
yhist = zeros(maxIter,1);
stoppedEarly = false;

fprintf('Iter| GradNorm | ObjFunc \n');
for k = 1:maxIter
    % step size or schedule
    ak = step(k);
    
    % update
    xc = xc + ak*d;

    % Descent direction for next step
    % Gradient Norm for stopping rule
    fc  = f(xc);
    d   = -g(xc);
    gnormc = sqrt(d'*d);
    yhist(k) = fc ;

    % Iteration Printing
    if printIters && ( mod(k,50)==0)  
        toDisp = [k, gnormc, fc ];
        fprintf('%3i | %06.3f | %08.3f \n', toDisp)
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
