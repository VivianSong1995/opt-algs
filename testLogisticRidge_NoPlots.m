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
[ ~, xs11 ] = newton_method_updt(x0, myhess, mygrad, myfunc, 140, 1e-4, false, true); % best!
opt_val = myfunc(xs11) ;
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
[ it11, xs11 ] = newton_method_updt(x0, myhess, mygrad, myfunc, 140, 1e-4, false, true); % best!

