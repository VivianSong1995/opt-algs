function [ out ] = logistic(t) 
%logistic(t) applies logistic function to elements in t   
%   logitstic(x) = exp(x)/( 1+exp(x) )
%# to prevent overflow we take the >0, <=0 case separately 
    n=length(t);
    idx = t > 0;
    out = zeros(n,1);
    out(idx) = 1 ./ (1 + exp(-1*t(idx) ) );
    %case ii
    exp_t = exp(t(~idx));
    out(~idx) = exp_t ./ (1 + exp_t);
end