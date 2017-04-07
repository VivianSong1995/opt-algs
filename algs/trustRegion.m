function [yhist, x] = trustRegion(x0,f,g,H,method,tol,maxit,toPlot,itprint)

% Initialize Algorithm Parameters
maxtrr = 8; %or 5 or min(norm(gc),10); % Maximum Trust Region Radius 
eta = 1/5;  %  eta in [0,1/4)
trrad = 4;  % Initial "Current" Trust Region Radius

% Algorithm Parameters
k = 1;
xc = x0;

% Initialize functional values
fc = f(xc);
gc = g(xc);
Hc = H(xc);

% Plotting
xx = 1:maxit;
yy = repmat(f(xc),1,maxit);

% Storage
yhist = zeros(maxIter,1);

if toPlot
    clf
    zz=semilogy(xx,yy,'YDataSource','yy');
end


fprintf('Iterate | radius | DeltaX | gradNorm  | objFunc   \n')
% Main While Loop
while(norm(gc) > tol && k <= maxit)
    %plotting
    yy(k) = fc;
    
    % Obtain pk by (approx) solving the TR subproblem
    if(method == 1)    % Method 1: Cauchy Point
    pk = cauchyPoint(gc,Hc,trrad);
    elseif(method==2)  % Method 2: Dogleg Method
    [pk,stepkey] = doglegMethod(gc,Hc,trrad);
    elseif(method==3)  % Method 3: 2D Subspace Method
    [pk,stepkey] = twoDimSubspace(gc,Hc,trrad);
    elseif(method==4)  % Method 4: Iterative More-Sorensen
    laminit = 1;
    pk = moreSorensen(gc,Hc,trrad,laminit);
    end
    
    % potential step info
    xn = xc + pk; % Potential new point
    fn = f(xn);
    gn = g(xn);
    Hn = H(xn); %modelHessian/Hessian
    mn = model(pk,fn,gn,Hn);
    
    % Evaluate rhoK
    rhok = relativeReduction(fc,fn,mn);
    
    % What do with TR Radius
    if(rhok < .25)
        trrad = .25*trrad;
    else
        if(rhok > .75 && norm(pk,2) == trrad) % plus tolerance
            trrad = min(2*trrad,maxtrr);
        %else trrad = trrad
        end
    end
    
    % What to do with point/step/direction
    if(rhok > eta)
        xc = xn;%xc + pk;
        fc = f(xc);
        gc = g(xc);
        Hc = H(xc);
    %else xc = xc
    end
    
    yhist(k) = fc ; % 
    
    if toPlot
        refreshdata(zz,'caller') 
        drawnow
    end
    
    % stopping rule
    if itprint; 
        if(method==2 || method==3 )
            %toDisp = [k, trrad, stepkey, norm(xn-xc,2), norm(gn,2), fc ];
            fprintf('Iter %03i| %5.4f | %5s | %06.3f | %09.3f | %09.3f \n', ...
                k, trrad, stepkey, norm(xn-xc,2), norm(gn,2), fc); %toDisp)
        else
            toDisp = [k, trrad, norm(xn-xc,2), norm(gn,2), fc ];
            fprintf('Iter %03i| %5.4f | %06.3f | %09.3f | %09.3f \n', toDisp)
        end
        %disp([k,round(norm(xc-x_old,2),4), f(xc,1:n) ]); 
    end
    
    k = k + 1;
end

if k <= maxit
    lastIter = k; 
    yhist = yhist(1:lastIter);
end

x = xc;
end
%
%
function [mk] = model(p,f,g,B) 
%
mk = f + g'*p + 1/2*p'*B*p;
end
%
%
function [rho] = relativeReduction(fc,fn,mn)
%
%   Input:  fc = current value of f
%           fn = new value of f
%           mn = new value of model
%
mz = fc; % model at zero
rho = (fc - fn)/(mz - mn);
end
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
function [pDog, stepFlag] = doglegMethod(g,B,trradc)
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
    stepFlag = 'steep';
% If Newton/Full step is inside TR, take it.  
elseif (norm(pB,2) <= trradc)
    pDog = pB;
    stepFlag = 'full ';
% Find intersection of dogleg path with boundary
else
    pBmU = pB - pU;
    aze = pBmU'*pBmU; bze = -2*pBmU'*pU; 
    cze = pU_norm*pU_norm - trradc*trradc;
    alphStar = (-bze+sqrt((bze*bze) - 4*aze*cze))/(2*aze);
    pDog = pU + alphStar*pBmU;
    stepFlag = 'bdry ';
end
end
%
function [pTwoD, stepFlag] = twoDimSubspace(g,B,trradc)
%
%
%
%gquadform = g'*B*g;
%pU = -(g'*g)/(gquadform)*g;
%pU_norm = norm(pU,2);
pB = -B\g; %full step

% If steepest descent step is outside of TR, scale back and go along.  
%if (pU_norm >= trradc)
%pTwoD = (trradc/pU_norm)*pU;
%stepFlag = 'steep';
%elseif
if(norm(pB,2) <= trradc)
pTwoD = pB;
stepFlag = 'full ';
else
stepFlag = 'bdry ';
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
%disp(r);
r = r(imag(r) == 0);
%disp(r);
%disp(Bt);
%disp(gt);

Z = trradc * [ (2*r)./(1+r.^2) ; (1-r.^2)./(1+r.^2)];
%disp(Z);
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