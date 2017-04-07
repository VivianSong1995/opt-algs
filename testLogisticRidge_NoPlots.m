% Regularized Logistic Ridge Regression
% Written by Arturo Fernandez 

% Setup
clear all
format long g

% Data
load('example_data.mat')
[n , p] = size(X);

% Train/Test
train_frac = 3/4 ;
nTrain = ceil(n * train_frac);
nTest = n - nTrain ;

rng(34); %set seed
idxTrain = randsample(n,nTrain);
idxSubVec = zeros(n,1);
idxSubVec(idxTrain) = 1 ;
idxSubVec = logical(idxSubVec);

XTrain = X(idxSubVec,:);
XTest = X(~idxSubVec,:);
yTrain = y(idxSubVec);
yTest = y(~idxSubVec);

% center
trainColMeans = mean(XTrain,1);
XTrain = XTrain - (ones(nTrain,1) * trainColMeans);
XTest = XTest - (ones(nTest,1) * trainColMeans);

% scale
trainColSDs = std(XTrain,0,1);
XTrain = XTrain ./ (ones(nTrain,1) * trainColSDs);
XTest = XTest ./ (ones(nTest,1) * trainColSDs);

XTrain = [ones(nTrain,1) XTrain];
XTest  = [ones(nTest,1) XTest];
p = p + 1 ;

%% Problem Paramater Initialization
lambda = 1e2 ;

%Def'ns
myfunc = @(w) logisticRidgeCostDivByN(w, XTrain, yTrain, lambda); 
mygrad = @(w) logisticRidgeGradientDivByN(w, XTrain, yTrain, lambda);
myhess = @(w) logisticRidgeHessianDivByN(w, XTrain, lambda);

myfX = @(w, X, y) logisticRidgeCostDivByN(w, X, y, lambda); 
mygX = @(w, X, y) logisticRidgeGradientDivByN(w, X, y, lambda);
myhessdiag = @(w) diag(diag(myhess(w),0));

% Initial guess and stepsizes
x0 = zeros(p,1);

% Baseline
[ yh17, w17 ] = newton_method_updt(x0, myhess, mygrad, myfunc, 140, 1e-4, false, true); % best!
opt_val = myfunc(w17) ;
err = @(x) x - opt_val ;

%% Gradient Method - Step Size Comparison
step1 = @(k) .001;
step2 = @(k) .001/k;
step3 = @(k) .01/sqrt(k);
step4 = @(k) 1/k^2;
eig_max = norm(XTrain,2);
step5 = @(k) (2/eig_max);

% Run
[ yh1, w1 ] = gradientDescent(x0, myfunc, mygrad, step1, 2e3, 1e-4, true); % ++ converges
[ yh2, w2 ] = gradientDescent(x0, myfunc, mygrad, step2, 2e3, 1e-4, true); % X - super slow
[ yh3, w3 ] = gradientDescent(x0, myfunc, mygrad, step3, 2e3, 1e-4, true); % ++ converges
[ yh4, w4 ] = gradientDescent(x0, myfunc, mygrad, step4, 2e3, 1e-4, true); % X - decay too fast
[ yh5, w5 ] = gradientDescent(x0, myfunc, mygrad, step5, 2e3, 1e-4, true); % X - too big

l1 = 1:length(yh1); l2 = 1:length(yh2); l3 = 1:length(yh3); 
l4 = 1:length(yh4); l5 = 1:length(yh5);

% Plot algorithm performance
f1 = figure;
figure(f1);
semilogy(l1,err(yh1), l2,err(yh2), l3,err(yh3), l4,err(yh4), l5,err(yh5));
legend('1e-3','k^{-1}','k^{-1/2}','k^{-2}','5e-3')
xlim([0 100])
ylim([1e-11 1e2]); 

%% Stochastic Gradient Method - Step Size Comparison
step1 = @(k) .001;
step2 = @(k) .001/k;
step3 = @(k) .01/sqrt(k);
step4 = @(k) 1/(k*k);
eig_max = norm(XTrain,2);
step5 = @(k) (2/eig_max)*1e-2; % too volatile otherwise
step6 = @(k) 1/(k + 1e5); 

% Run
numSteps = 3e3;
[ yh6, w6 ]   = stochasticGradientDescent(x0, XTrain, yTrain, 32, myfunc, mygX, step1, numSteps, 1e-4, true); % no
[ yh7, w7 ]   = stochasticGradientDescent(x0, XTrain, yTrain, 32, myfunc, mygX, step2, numSteps, 1e-4, true); % slow but converging
[ yh8, w8 ]   = stochasticGradientDescent(x0, XTrain, yTrain, 32, myfunc, mygX, step3, numSteps, 1e-4, true); % no
[ yh9, w9 ]   = stochasticGradientDescent(x0, XTrain, yTrain, 32, myfunc, mygX, step4, numSteps, 1e-4, true); % slow  but converging
[ yh10, w10 ] = stochasticGradientDescent(x0, XTrain, yTrain, 32, myfunc, mygX, step5, numSteps, 1e-4, true); % no
[ yh11, w11 ] = stochasticGradientDescent(x0, XTrain, yTrain, 32, myfunc, mygX, step6, numSteps, 1e-4, true); % no

st6 = 1:numSteps;
f2 = figure;
figure(f2);
semilogy(st6,err(yh6), st6,err(yh7), st6,err(yh8), st6,err(yh9),st6,err(yh10), st6,err(yh11) ); 
ylim([0 3e-3])
legend('1e-3','k^{-1}','k^{-1/2}','k^{-2}','5e-5','1/(k+1e5)')

%% Newton's Method
step1 = @(k) .001;
step2 = @(k) .001/k;
step3 = @(k) .01/sqrt(k);
step4 = @(k) 1/k^2;
eig_max = norm(XTrain,2);
step5 = @(k) (2/eig_max);

% Run
numSteps = 10;
[ yh12, w12 ] = newton_method(x0, myhess, mygrad, myfunc, step1, numSteps, 1e-4, false, true); % meh
[ yh13, w13 ] = newton_method(x0, myhess, mygrad, myfunc, step2, numSteps, 1e-4, false, true); % meh
[ yh14, w14 ] = newton_method(x0, myhess, mygrad, myfunc, step3, numSteps, 1e-4, false, true); % meh
[ yh15, w15 ] = newton_method(x0, myhess, mygrad, myfunc, step4, numSteps, 1e-4, true, true); % great!
[ yh16, w16 ] = newton_method(x0, myhess, mygrad, myfunc, step5, numSteps, 1e-4, false, true); % meh 

l12 = 1:length(yh12); l13 = 1:length(yh13); l14 = 1:length(yh14); 
l15 = 1:length(yh15); l16 = 1:length(yh16);

%semilogy(l12,err(yh12), l13,err(yh13), l14,err(yh14), l15,err(yh15), l16,err(yh16)); 
%ylim([0 3e-3])
%legend('1e-3','k^{-1}','k^{-1/2}','k^{-2}','5e-3')
% step4 converged in 1 step.

% Newton Method with BT Line Search and default BT parameters
% Overall the standard !
%[ yh17, w17 ] = newton_method_updt(x0, myhess, mygrad, myfunc, 140, 1e-4, false, true); % best!

%% Trust Region
% Cauchy
[ yh18, w18 ] = trustRegion(x0, myfunc, mygrad, myhess, 1, 1e-4, 160, false, true); % okay

% Dogleg
[ yh19, w19 ] = trustRegion(x0, myfunc, mygrad, myhess, 2, 1e-4, 140, false, true); % amazing!
[ yh20, w20 ] = trustRegion(x0, myfunc, mygrad, myhessdiag, 2, 1e-4, 140, false, true); % slower

%  More-Sorensen (iterative)
% fails if you start at 0
[ yh21, w21 ] = trustRegion(x0+1, myfunc, mygrad, myhess, 4, 1e-4, 140, false, true); % much slower

% Two-Dimensional Subspace Minimization
[ yh22, w22 ] = trustRegion(x0, myfunc, mygrad, myhess, 3, 1e-4, 60, false, true); % awesome!
[ yh23, w23 ] = trustRegion(x0, myfunc, mygrad, myhessdiag, 3, 1e-4, 100, false, true); % slower


l18 = 1:length(yh18); l19 = 1:length(yh19); l20 = 1:length(yh20); 
l21 = 1:length(yh21); l22 = 1:length(yh22); l23 = 1:length(yh23); 

f3 = figure;
figure(f3);
semilogy(l18,err(yh18), l19,err(yh19), l20,err(yh20), l21,err(yh21), ...
         l22,err(yh22), l23,err(yh23)); 
legend('Cauchy','Dog-FullH','Dog-DiagH','More-Sor','2DSS-FullH','2DSS-DiagH')
% 2D Subspace converged in 1 step.


%% BFGS
tmp_diag = diag(XTrain'*XTrain);
tmp_diag(tmp_diag==0) = 1;
inv_diag = round(1./tmp_diag,4);
H01 = diag(tmp_diag);
H02 = sparse(diag(inv_diag));
H03 = sparse(diag(inv_diag)) + 1e-3*speye(p);
H04 = sparse(diag(inv_diag)) + speye(p);

[ yh30, w30 ] = bfgs(x0, myfunc, mygrad, H01, 1e-4, 200, true, false,true); 
%[ yh31, w31 ] = bfgs(x0, myfunc, mygrad, H02, 1e-3, 200, true, false, true); % X - gets stuck in line search
[ yh32, w32 ] = bfgs(x0, myfunc, mygrad, H03, 1e-4, 200, true, false, true); % works best, inv. diag. + small I
[ yh33, w33 ] = bfgs(x0, myfunc, mygrad, H04, 1e-4, 200, true, false, true); % inverse diagonal + I
[ yh34, w34 ] = bfgs(x0, myfunc, mygrad, speye(p), 1e-4, 200, true, false, true); % identity

% These juse use backtracking as opposed to Strong Wolfe (faster iterations)
[ yh35, w35 ] = bfgs(x0, myfunc, mygrad, H01, 1e-4, 200, false, true, true); % 
% [ yh36, w36 ] = bfgs(x0, myfunc, mygrad, H02, 1e-3, 200, false, true, true); % fails
[ yh37, w37 ] = bfgs(x0, myfunc, mygrad, H03, 1e-4, 200, false, true, true); % does worse than stronge wolfe but faster iters 
[ yh38, w38 ] = bfgs(x0, myfunc, mygrad, H04, 1e-4, 200, false, true, true); % was 
[ yh39, w39 ] = bfgs(x0, myfunc, mygrad, speye(p), 1e-4, 200,false, true, true); % was 

l30 = 1:length(yh30); l32 = 1:length(yh32); l33 = 1:length(yh33); l34 = 1:length(yh34);
l35 = 1:length(yh35); l37 = 1:length(yh37); l38 = 1:length(yh38); l39 = 1:length(yh39);

f4 = figure;
figure(f4);
semilogy(l30,err(yh30),'-+k',  l32,err(yh32),'-+b',  l33,err(yh33),'-+g',  l34,err(yh34),'-+r', ...
         l35,err(yh35),'--oy', l37,err(yh37),'--or', l38,err(yh38),'--*m', l39,err(yh39),'--oc'); 
legend('SW-Diag', 'SW-RegInvDiag','SW-InvDiag','SW-Id', ...
       'BT-Diag', 'BT-RegInvDiag','BT-InvDiag','BT-Id')
% 2D Subspace converged in 1 step.

%% sr1-trust
tmp_diag2 = diag(XTrain'*XTrain);
B01 = diag(tmp_diag2) + 1e-3*speye(p);
[ yh40, w40 ] = sr1Trust(x0, B01, myfunc, mygrad, 2, 1e-4, 200, false, true); 
[ yh41, w41 ] = sr1Trust(x0, speye(p), myfunc, mygrad, 2, 1e-4, 200, false, true); 
% fast iterations!!!

%% bfgsTrust
%B01 = inv(H01);
B03 = inv(H03); % had worked best before, so we'll stick with that.
%B04 = inv(H04);

% Cauchy
[ yh50, w50 ] = bfgsTrust(x0, myfunc, mygrad, B03, H03, 1, 1e-4, 200, 1, true, true); %breaks
[ yh51, w51 ] = bfgsTrust(x0, myfunc, mygrad, B03, H03, 1, 1e-4, 200, 2, true, true); %pos definiteness not guaranteed
[ yh52, w52 ] = bfgsTrust(x0, myfunc, mygrad, B03, H03, 1, 1e-4, 200, 3, true, true); %solid

% Dogleg
[ yh53, w53 ] = bfgsTrust(x0, myfunc, mygrad, B03, H03, 2, 1e-4, 200, 1, true, true); 
[ yh54, w54 ] = bfgsTrust(x0, myfunc, mygrad, B03, H03, 2, 1e-4, 200, 2, true, true);
[ yh55, w55 ] = bfgsTrust(x0, myfunc, mygrad, B03, H03, 2, 1e-4, 200, 3, true, true);

% 2D Subspace Min
[ yh56, w56 ] = bfgsTrust(x0, myfunc, mygrad, B03, H03, 3, 1e-4, 200, 1, true, true); 
[ yh57, w57 ] = bfgsTrust(x0, myfunc, mygrad, B03, H03, 3, 1e-4, 200, 2, true, true);
[ yh58, w58 ] = bfgsTrust(x0, myfunc, mygrad, B03, H03, 3, 1e-4, 200, 3, true, true);

% Skip More-Sorensen (positive definiteness constraints)


