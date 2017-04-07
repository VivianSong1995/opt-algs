function [yhist, x] = bfgsTrust(x0,f,g,B0,H0,method,tol,maxit,btFlag,toPlot,itprint)%,reso)
%function [x,histout,costdata] = trustRegion(x0,f,g,h,tol,maxit)%,reso)

% Output: x = solution
%         histout = iteration history   
%             Each row of histout is
%       [norm(grad), f, TR radius, iteration count] 
%         costdata = [num f, num grad, num hess] 

% Initialize Algorithm Parameters
maxtrr = 8; %5; % =min(norm(gc),10); % Maximum Trust Region Radius 
eta = 1/5;  % 1/4; eta in [0,1/4)
trrad = 4;  % Initial "Current" Trust Region Radius

% Algorithm Parameters
k=1;
xc=x0;

% Initialize functional values
fc = f(xc); 
gc = g(xc);
Hc = H0;
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

fprintf('Iterate | radius | DeltaX | gradNorm  | objFunc   \n')
% Main While Loop
while(norm(gc) > tol && k <= maxit)
    %plotting
    yy(k)=fc;
    % should have B/H
    
    % Obtain pk by (approx) solving the TR subproblem
    if(method == 1)    % Method 1: Cauchy Point
    dc = cauchyPoint(gc,Bc,trrad);
    elseif(method==2)    % Method 2: Dogleg Method --- Replace Full step with (H)^(-1) info
    dc = doglegMethod(gc,Bc,Hc,trrad);
    elseif(method==3)    % Method 3: 2D Subspace Method
    dc = twoDimSubspace(gc,Bc,Hc,trrad);
    elseif(method==4)    % Method 4: Iterative More-Sorensen -- (B_lam*I)^(-1) step
    laminit = 1;
    dc = moreSorensen(gc,Bc,trrad,laminit);
    end
    
    % step size
    if(btFlag == 1)
        ac = strongwolfe(f,g,dc,xc,fc,gc,1,10);
    elseif(btFlag == 2)
        ac = btLineSearch(f,fc,xc,gc,dc);
    else
        ac = 1;
    end
    
    % potential step info, update info, and curvature info (sc/yc)
    sc = ac*dc;
    xn = xc + sc; % Potential new point
    fn = f(xn);
    gn = g(xn);
    yc = gn - gc;
        
    % Computation of curvature info and relative reduction
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
            trrad = min(2*trrad,maxtrr);%2*trrad; %min(2*trrad,maxtrr);
        %else (sk_norm <= 0.8*trrad) %trrad = trrad;
        end
    elseif (0.1 <= rred) && (rred <= 0.75)
        trrad = trrad; %#ok<ASGSL>
    else
        trrad = 0.5*trrad;
    end
    
    % What to do with Hessian Approximation
    % Compute New Approx Inverse Hessian
    r = 1/(sc'*yc);
    syTrH = sc*(yc'*Hc);
    Hc = Hc - r*(syTrH+syTrH')  + ((r*r)*(yc'*Hc*yc) + r)*(sc*sc');
    % Compute New Approx Hessian
    Bs = Bc*sc;
    Bc = Bc - (1/(sc'*Bc*sc))*(Bs*Bs')  + r*(yc*yc');
    
    %if hessUpdate
    %else
    %resc = yc - Bc*sc;
    %rds = resc'*sc;
    %resc_norm = norm(resc,2);
    %if( abs(rds) >= r * sc_norm * resc_norm)
    %Bc = Bc + (1/rds)*(resc*resc');
    %else Bc = Bc
    %end
    %end

    % PLotting
    if toPlot
        refreshdata(zz,'caller'); drawnow;
    end
    if itprint; 
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
%function [mk] = model(p,f,g,B) 
%%
%mk = f + g'*p + 1/2*p'*B*p;
%end
%
%
%function [rho] = relativeReduction(fc,fn,mn)
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
function [pDog] = doglegMethod(g,B,H,trradc)
%
% B is approx Hessian
% H is approx inverse Hessian
%
gquadform = g'*B*g;
pU = -(g'*g)/(gquadform)*g;
pU_norm = norm(pU,2);
pB = -H*g; %full step

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
function [pTwoD] = twoDimSubspace(g,B,H,trradc)
%
%
%
%gquadform = g'*B*g;
%pU = -(g'*g)/(gquadform)*g;
%pU_norm = norm(pU,2);
pB = -H*g; %full step

% If steepest descent step is outside of TR, scale back and go along.  
%if (pU_norm >= trradc)
%pTwoD = (trradc/pU_norm)*pU;
%stepFlag = 'steep';
%elseif
if(norm(pB,2) <= trradc)
pTwoD = pB;
%stepFlag = 'full ';
else
%stepFlag = 'bdry ';
S = [g pB];
[Q,~] = qr(S,0);
%disp(Q);
Bt = Q'*B*Q;
gt = Q'*g;

a = Bt(1,1) * trradc*trradc;
b = Bt(1,2) * trradc*trradc;
c = Bt(2,2) * trradc*trradc;

d = gt(1) * trradc;
f = gt(2) * trradc;

coefs = [ -b +d , 2*(a-c+f) , 6*b , 2*(-a+c+f) , -b-d]';
r = roots(coefs)';
r = r(imag(r) == 0);


Z = trradc * [ (2*r)./(1+r.^2) ; (1-r.^2)./(1+r.^2)];
vals = 0.5*diag(Z'*Bt*Z) + Z'*gt;
[~,I] = min(vals);
w = Z(:,I);
pTwoD = Q*w;
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