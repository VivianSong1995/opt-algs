function [ yhist, xc ] = bfgs(x0, f,g,H0, tol, maxit, swFlag, toPlot, itprint)
% Code by Arturo Fernandez, Winter 2015

%# Gradient of the logistic loss

% Initialization
if nargin < 4
tol=1e-4;
end
if nargin < 5
maxit = 500;
end
if ~exist('toPlot','var');   toPlot = false;      end
if ~exist('itprint','var');   itprint = false;      end

% Algorithm Parameters
k = 1;
xc = x0;

% Initialize functional values
fc = f(xc);
gc = g(xc);
Hc = H0;

% Plotting
xx = 1:maxit;
yy = repmat(fc,1,maxit);

if toPlot
    clf; zz=semilogy(xx,yy,'YDataSource','yy');
end

%storage
yhist = zeros(maxit, 1);
yhist(1) = fc ;


fprintf('Iterate | DeltaX | gradNorm  | objFunc   \n')
% Main While Loop
while(norm(gc) > tol && k <= maxit)
    % search direction
    d = -Hc*gc;
    % step size
    if swFlag
        ac = strongwolfe(f,g,d,xc,fc,gc,10,1); %amax > 1, 10 works
    else
        ac = btLineSearch(f,fc,xc,gc,d);
    end
    % update
    xn = xc + ac*d;
    gn = g(xn);
    
    % Curvature Info
    sc = xn - xc;
    yc = gn - gc;
    
    % Compute New Approx Hessian
    r = 1/(sc'*yc);
    syTrH = sc*(yc'*Hc);
    Hc = Hc - r*(syTrH+syTrH')  + ((r*r)*(yc'*Hc*yc) + r)*(sc*sc');
    
    % Update Ticker
    k = k + 1;
    
    %% Plotting
    xc = xn;
    fc = f(xc);
    gc = gn;
    yy(k) = fc;
    yhist(k) = fc ;
    
    % Plot update
    if toPlot
        refreshdata(zz,'caller') 
        drawnow
    end
    
    if itprint; 
        toDisp = [k, norm(sc,2), norm(gn,2), fc ];
        fprintf('Iter %03i| %06.3f | %09.3f | %09.3f \n', toDisp)
    end
end

if k <= maxit
    lastIter = k;%-1; 
    yhist = yhist(1:lastIter);
end

end
