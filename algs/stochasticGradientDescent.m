function [ yhist, xc ] = stochasticGradientDescent(x0,X,y,mbsz,f,g,step,maxIter,tol,printIters) 
%# Gradient of the logistic loss

% ------------Initialization----------------
if ~exist('maxIter','var');     maxIter = 1e6;      end
if ~exist('tol','var');         tol = 1e-4;           end
if ~exist('printIters','var');  printIters = false;   end

[n,~] = size(X);

gBlkIdx = randperm(n,mbsz); 
Xtmp = X(gBlkIdx,:);
ytmp = y(gBlkIdx);

% initialization
xc = x0;
d  = -g(xc, Xtmp, ytmp);
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
    %
    gBlkIdx = randperm(n,mbsz); 
    Xtmp = X(gBlkIdx,:);
    ytmp = y(gBlkIdx);
    
    fc  = f(xc);
    d  = -g(xc, Xtmp, ytmp);
    dreal = -g(xc, X, y);
    gnormc = sqrt(dreal'*dreal);
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
