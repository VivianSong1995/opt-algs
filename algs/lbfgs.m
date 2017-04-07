function [ yhist, xc ] = lbfgs(x0,m,f,g,H0,tol,maxit,swFlag,toPlot,itprint)
%# Gradient of the logistic loss

% Initialization
if nargin < 4
tol=1e-4;
end
if nargin < 5
maxit = 500;
end
if ~exist('toPlot','var');   toPlot=false;      end
if ~exist('itprint','var');   itprint=false;      end

% Algorithm Parameters
k=0;
xc=x0;
p = length(x0);

% Initialize functional values
fc = f(xc);
gc = g(xc);
Hc0 = H0;

% PLotting
xx=1:maxit;
yy=repmat(fc,1,maxit);

% Pairs
S = zeros(p,m);
Y = zeros(p,m);
rho = zeros(1,m);

if toPlot
    clf; zz=semilogy(xx,yy,'YDataSource','yy');
end

%storage
yhist = zeros(maxit, 1);
yhist(1) = fc ;

fprintf('Iterate | DeltaX | gradNorm  | objFunc   \n')
% Main While Loop
while(norm(gc) > tol && k <= maxit)
    % Choose Hk0 (will actually be done at the end of this iteration)
    
    % Compute pk <- -Hk*gk using twoLop
    %dc = twoLoop(k,m,gc,rho,S,Y,Hc0);
    %disp(m)
    %dc = (-1)*twoLoop(k,m,gc,rho,S,Y,Hc0);
    dc = (-1)*twoLoop(m,gc,rho,S,Y,Hc0);
    %if(gc'*dc > 0)
    %    dc=-dc;
    %end just loops
    %disp([k, gc'*dc, norm(dc,2)])
    % step size

    if swFlag
       ac = strongwolfe(f,g,dc,xc,fc,gc,10,1); %amax > 1, 10 works
    else
       ac = btLineSearch(f,fc,xc,gc,dc);
    end
    
    % update steps/gradients and curvature (sc/yc)
    sc = ac*dc;
    xn = xc + sc;
    gn = g(xn);
    yc = gn - gc;
    
    % Replace 1:(m-1) columns with old 2:m
    %if(k > m)
    S(:,1:(m-1)) = S(:,2:m);
    Y(:,1:(m-1)) = Y(:,2:m);
    rho(1:(m-1)) = rho(2:m);
    %end
    
    % Compute new
    S(:,m) = sc;
    Y(:,m) = yc;
    rho(m) = 1/(yc'*sc);
    
    % Update Initialization of HEssian
    gammac = (sc'*yc)/(yc'*yc);
    Hc0 = gammac*speye(p); % may be made sparse
    
    % Update Ticker
    k = k + 1;
    
    %% Plotting
    xc = xn;
    fc = f(xc);
    gc = gn;
    yhist(k+1) = fc ;
    
    if (k <= maxit && toPlot)
        yy(k)=fc;    
        % Plot update
        refreshdata(zz,'caller') 
        drawnow
    end
    
    if itprint; 
        toDisp = [k, norm(sc,2), norm(gn,2), fc ];
        fprintf('Iter %03i| %06.3f | %09.3f | %09.3f \n', toDisp)
    end
end
iters = k-1;

if iters < maxit
    clf;
    semilogy(xx(1:iters),yy(1:iters),'YDataSource','yy');
end

if k <= maxit
    lastIter = k+1;%
    yhist = yhist(1:lastIter);
end

end


    

%% LBFGS Two-Loop Recursion (k,m
%function [r] = twoLoop(k,m,gc,rho,S,Y,Hc0) %iterative
function [r] = twoLoop(m,gc,rho,S,Y,Hc0) %iterative
%
q = gc;
a = zeros(1,m);

%backMax = min(m-1,k);
%fwdMax = min(m,k+1);

for i = 0:(m-1)%backMax%(m-1) %1:m
    a(m-i)= rho(m-i)*S(:,m-i)'*q;
    q = q - a(m-i)*Y(:,m-i);
end
r = Hc0*q;
for i = 1:m%fwdMax%m
    b = rho(i)*Y(:,i)'*r;
    r = r + S(:,i)*(a(i) - b);
end
end
