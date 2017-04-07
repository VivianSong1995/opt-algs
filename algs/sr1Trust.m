function [yhist, x] = sr1Trust(x0,B0,f,g,method,tol,maxit,toPlot,itprint)%,reso)
%function [x,histout,costdata] = trustRegion(x0,f,g,h,tol,maxit)%,reso)

% Output: x = solution
%         histout = iteration history   
%             Each row of histout is
%       [norm(grad), f, TR radius, iteration count] 
%         costdata = [num f, num grad, num hess] 

% Initialize Algorithm Parameters
maxtrr = 8; %5; % =min(norm(gc),10); % Maximum Trust Region Radius 
eta = 1e-4; % 1e-4; eta in (0,1e-3)
trrad = 4;  % Initial "Current" Trust Region Radius
r = 1e-8;   % tiny r in (0,1)

% Algorithm Parameters
k=1;
xc=x0;

% Initialize functional values
fc = f(xc);
gc = g(xc);
Bc = B0;

% PLotting
xx = 1:maxit;
yy = repmat(f(xc),1,maxit);

%storage
yhist = zeros(maxit, 1);
yhist(1) = fc ;

if toPlot
    clf; zz=semilogy(xx,yy,'YDataSource','yy');
end


fprintf('Iterate | rad | DeltaX | gradNorm  | objFunc   \n')
% Main While Loop
while(norm(gc) > tol && k <= maxit)
    %plotting
    yy(k)=fc;
    
    % Obtain sk by (approx) solving the TR subproblem
    if(method == 1)
    % Method 1: Cauchy Point
    sc = cauchyPoint(gc,Bc,trrad);
    elseif(method==2)
    % Method 2: Dogleg Method
    sc = doglegMethod(gc,Bc,trrad);
    elseif(method==3)
    % Method 3: 2D Subspace Method
    %pk = doglegMethod(gc,Hc,trrad);
    %elseif(method==4)
    % Method 4: Iterative More-Sorensen -  can't use since 
    % B_{k+1} not guaranteed to be PD
    %laminit = 1;
    %sc = moreSorensen(gc,Bc,trrad,laminit);
    end
    
    % potential step info
    xn = xc + sc; % Potential new point
    fn = f(xn);
    gn = g(xn);
    
    % Computation of curvature info and relative reduction
    yc = gn - gc;
    ared = fc - fn;
    pred = -(gc'*sc + 1/2*sc'*Bc*sc);
    rred = ared/pred;
    
    % What to do with point/step/direction
    if(rred > eta)
        xc = xn;
        fc = f(xc);
        gc = g(xc);
    %else xc = xc;
    end
    
    sc_norm = norm(sc,2);
    % What do with TR Radius
    if(rred > 0.75)
        if(sc_norm > 0.8*trrad)
            trrad = 2*trrad;%min(2*trrad,maxtrr);
        %else (sk_norm <= 0.8*trrad) %trrad = trrad;
        end
    elseif (0.1 <= rred) && (rred <= 0.75)
        trrad = trrad; %#ok<ASGSL>
    else
        trrad = 0.5*trrad;
    end

    % What to do with Hessian Approximation
    resc = yc - Bc*sc;
    rds = resc'*sc;
    resc_norm = norm(resc,2);
    if( abs(rds) >= r * sc_norm * resc_norm)
        Bc = Bc + (1/rds)*(resc*resc');
    %else Bc = Bc
    end
    
    % PLotting
    if toPlot
    refreshdata(zz,'caller'); drawnow;
    end
    
    if itprint
        toDisp = [k, trrad, norm(xn-xc,2), norm(gn,2), fc ];
        fprintf('Iter %03i|%4.3f | %06.3f | %09.3f | %09.3f \n', toDisp)
    end
    
    k = k + 1;
    yhist(k) = fc ;
end

x = xc;

if k <= maxit
    lastIter = k;%-1; 
    yhist = yhist(1:lastIter);
end

end
%
%
% function [mk] = model(p,f,g,B) 
% %
% mk = f + g'*p + 1/2*p'*B*p;
% end
% %
% %
% function [rho] = relativeReduction(fc,fn,mn)
% %
% %   Input:  fc = current value of f
% %           fn = new value of f
% %           mn = new value of model
% %
% mz = fc; % model at zero
% rho = (fc - fn)/(mz - mn);
% end
%
%
function [pC] = cauchyPoint(g,B,trradc) 
%
%
%
gquadform = g'*B*g;
gnorm = norm(g,2);

if (gquadform <= 0)
    tau = 1;
else
    tau = min(1, (gnorm^3)/(trradc* gquadform));
end

pC = (-tau)*(trradc/gnorm)*g;
end
%
function [pDog] = doglegMethod(g,B,trradc)
%
%
%
gquadform = g'*B*g;
pU = -(g'*g)/(gquadform)*g;
pU_norm = norm(pU,2);
pB = -B\g; %full step

% If steepest descent step is outside of TR, scale back and go along.  
if (pU_norm >= trradc)
    pDog = (trradc/pU_norm)*pU;
% If Newton/Full step is inside TR, take it.  
elseif (norm(pB,2) <= trradc)
    pDog = pB;
% Find intersection of dogleg path with boundary
else
    pBmU = pB - pU;
    aze = pBmU'*pBmU; bze = -2*pBmU'*pU; 
    cze = pU_norm*pU_norm - trradc*trradc;
    alphStar = (-bze+sqrt((bze*bze) - 4*aze*cze))/(2*aze);
    pDog = pU + alphStar*pBmU;
end
end
%
function [pIter] = moreSorensen(g,B,trradc,lam0) %iterative
%
%
%
[m,n]=size(B); %#ok<ASGLU>
%tol = 1e-3;
maxit = 3;
%k = 1;
lam = lam0;

I=speye(n);

%while(k <= maxit) [and tol in case cheaper]
for k = 1:maxit
    R = chol(B+lam*I);
    pc = R\(R'\(-g));
    qc = R'\pc;
    
    pc_norm = norm(pc,2);
    nr = pc_norm/norm(qc,2);
    lam = lam + (nr*nr)*((pc_norm - trradc)/trradc);
%    k = k+1;
end

pIter = -(B+lam*I)\g;
end