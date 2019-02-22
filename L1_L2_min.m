%Mixed L1 + L2 minimization using quadratic / linear programming
%   
%   x=L1_L2_min(M1,c1,M2,c2,A,b,Aeq,beq,lb,ub,options) 
%   solves the mixed L1 + L2 problem:
%
%   sum(abs(M1*x - c1)) + sum((M2*x-c2).^2)
%
%       subject  to  A*x <= b,  Aeq*x = beq,  lb<=x,  x<=ub 
%
%
%   Input arguments M2, c2, A, b, Aeq, beq, lb, ub, options are optional.
%   If M2 is not given or is equal to zero, the algorithm uses linear 
%   programming. Otherwise, it uses quadratic programming.
%
%   This solver decomposes the term M1*x-c1 into a positive and a negative 
%   part and then uses the quadratic (linear) programming algorithm 
%   "quadprog" ("linprog") to solve the L1 + L2 problem. The argument 
%   "options" determines the options used for the "quadprog" solver, or the
%   "linprog" solver, if M2 is not given as an input or is equal to zero.
%
%   Note: Defining M1, c1, M2, and c2 accordingly, the solver can be used
%         to implement Elastic Net, Lasso, etc. 
%
%Marco Menner (marco.menner88[at]gmail.com)


function x=L1_L2_min(M1,c1,M2,c2,A,b,Aeq,beq,lb,ub,options)



if nargin<11
    options=[];
end

n=size(M1,2);
if ~exist('lb','var')
    b=-inf*ones(n,1);
elseif isempty(lb)
    lb=-inf*ones(n,1);
end


if ~exist('ub','var')
    ub=inf*ones(n,1);
elseif isempty(lb)
    ub=inf*ones(n,1);
end

%Introduce new variables wp = max(M1*x-c1,0)  and  wn = -min(M1*x-c1,0).
%
%This gives  wp + wn = abs(M1*x-c1) and  wp - wn = M1*x-c1
%
%Then minimize  sum(wp+wn) + x'*M2'*M2*x - 2*c2'*M2*x  subject to:
%
%  wp-wn-M1*x = -c1,  wp>=0,  wn>=0, A*x <= b,  Aeq*x = beq,  lb<=x,  x<=ub
%
%and solve for wp, wn, and x


%Add condition: wp-wn-M1*x = -c1,
if ~exist('Aeq','var')
    Aeq_full=sparse([-M1,eye(n),-eye(n)]);
elseif isempty(Aeq)
    Aeq_full=sparse([-M1,eye(n),-eye(n)]);
else
    Aeq_full=sparse([Aeq,zeros(size(Aeq,1),2*n);...
        -M1,eye(n),-eye(n)]);
end


if ~exist('beq','var')
    beq=[];
end


if ~exist('A','var')
    A_full=[];
elseif isempty(A)
    A_full=[];
else
    A_full=sparse([A,zeros(size(A,1),2*n)]);
end


beq_full=[beq;-c1];

%Set condition wp >= 0  and  wn >= 0
lb_full=sparse([lb;zeros(2*n,1)]);
ub_full=[ub;inf(2*n,1)];


if isempty(M2) || any(any(M2))==0
    %Define f such that sum over wp + wn
    f=sparse([zeros(n,1);ones(2*n,1)]);
    x_help=linprog(f,A_full,b,Aeq_full,beq_full,lb_full,ub_full,options);
else
    %Define f such that it includes sum over wp + wn and the linear part 
    %of L2 norm that includes x
    
    f=sparse([-2*M2'*c2;ones(2*n,1)]);
    H=sparse(blkdiag(M2'*M2,zeros(2*n)));
    x_help=quadprog(H*2,f,A_full,b,Aeq_full,beq_full,lb_full,ub_full,options);
end

x=x_help(1:n);


end