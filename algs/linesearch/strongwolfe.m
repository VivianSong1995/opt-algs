function astar = strongwolfe(f,g,d,x0,f0,g0,amax,qnFlag)
% function alphas = strongwolfe(f,d,x0,alpham)
% Line search algorithm satisfying the stron Wolfe conditions
% Algorithms 3.5 on pages 60-61 in Nocedal and Wright
% MATLAB Code by Karik Sivaramkrishnan
% Last modified: January 27, 2008
% Re-used by Arturo Fernandez, Winter 2015

% Algorithm PAramaters and Initialization
a0 = 0;
%a0 = 1;

c1 = 1e-4;
c2 = 0.9;%5;%0.9;%0.5; %.9 recommended by NW

ac = a0;
an = amax*rand(1);%.66;%rand(1);
%an = 1;
% ac is alpha_{i-1}
% an is alpha_i

% Defining phi
phi = @(a) f(x0 + a*d);
phiprime = @(a) g(x0 + a*d)'*d;

% intinialziation
phiprime0 = g0'*d;
phic = f0;

if qnFlag && ... 
   ( f(x0+d) <= phic+c1*phiprime0 ) && ...
   ( abs(g(x0+d)'*d) <= c2*abs(phiprime0) )
    astar = 1;
    return
end

i=1;
while 1
    phin = phi(an);    
    % Evaluate Phi of alpha_i
    if (phin > f0 + c1*an*phiprime0) || ((phin >= phic) && (i>1))
        astar = zoom(phi,phiprime,f0,phiprime0,ac,an);%,1);
        return
    end
    % Evaluate Phi Prime of alpha_i
    phiprimen = phiprime(an);
    if abs(phiprimen) <= -c2*phiprime0
        astar = an;
        return
    end
    if phiprimen >= 0
        astar = zoom(phi,phiprime,f0,phiprime0,an,ac);%,1);
        return
    end
    
    ac = an;
    phic = phi(ac);
    an = an + (amax-an)*.5;%rand(1);%.5;%rand(1);
    i = i+1;
end
end

function astar = zoom(phi,phiprime,phi0,phiprime0,alo,ahi)%,method)
% function alphas = zoom(f,x0,d,alphal,alphah)
% Algorithm 3.6 on pge 61 in Nocedal and Wright
% MATLAB Code by Kartik Sivaramakrishnan
% Last modified: January 27,2008

c1 = 1e-4;
c2 = 0.9;%5;%0.9;%0.5;

% Defining phi
%phi = @(a) f(xc + a*d);
%phiprime = @(a) g(xc + a*d)'*d;

while 1
    %Interpolation = rightno bisection
    %if(method == 1)
    aj = 1/2*(alo+ahi); 
    %else
    %aj = cubic(phi,phi0,phiprime0,alo,ahi);
    %disp(1)
    %end
    
    %Evaluation of Phi of alpha_j
    phij = phi(aj);
    philo = phi(alo);
   
    if ((phij > phi0 + c1*aj*phiprime0) || (phij >= philo)) 
        ahi = aj;
    else
        % Evaluate Phi Prime of alpha_j
        phiprimej = phiprime(aj);
        if abs(phiprimej) <= -c2*phiprime0
            astar = aj;
            return
        end
        if phiprimej*(ahi-alo) >= 0
            ahi = alo;
        end
        alo = aj;
    end
end
end

% function anext = cubic(phi,phi0,phiprime0,a0,a1)
%     a02 = a0*a0;
%     a12 = a1*a1;
%     z = 1/(a02*a12*(a1-a0)) * [ a02 -a12; -a0*a02 a1*a12] * ...
%         [ phi(a1) - phi0 - phiprime0*a1; ...
%           phi(a0) - phi0 - phiprime0*a0];
%     a = z(1); b = z(2);
%     anext = 1/(3*a) * (-b + sqrt(b*b - 3*a*phiprime0));
%     if (anext < min(a0,a1)) || (anext > max(a0,a1))
%         anext = (a0+a1)/2;
%     end    
% end
