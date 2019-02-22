%L1 minimization using linear programming
%   x=L1_min(M,c,A,b,Aeq,beq,lb,ub,options) solves the L1 problem
%
%   sum(abs(M*x - c))   subject  to  A*x <= b,  Aeq*x = beq,  lb<=x,  x<=ub 
%
%   The input arguments A, b, Aeq, beq, lb, ub, options are optional. 
%   This solver decomposes the term M*x-c into a positive and a negative 
%   part and then uses the linear programming algorithm "linprog" to solve
%   the L1 problem. The argument "options" determines the options used for 
%   the "linprog" solver.
%
%Developed by: Marco Menner (marco.menner88@gmail.com)


function x=L1_min(M,c,A,b,Aeq,beq,lb,ub,options)

n=size(M,2);

if nargin<9
    options=[];
end


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

%Introduce new variables wp = max(M*x-c,0)  and  wn = -min(M*x-c,0).
%
%This gives  wp + wn = abs(M*x-c) and  wp - wn = M*x-c
%
%Then minimize  sum(wp+wn)  s.t.  wp-wn-M*x = -c,  wp>=0,  wn>=0,
%
%                                 A*x <= b,  Aeq*x = beq,  lb<=x,  x<=ub
%
%and solve for wp, wn, and x


%Add condition: wp-wn-M*x = -c,
if ~exist('Aeq_full','var')
    Aeq_full=sparse([-M,eye(n),-eye(n)]);
elseif isempty(Aeq)
    Aeq_full=sparse([-M,eye(n),-eye(n)]);
else
    Aeq_full=sparse([Aeq,zeros(size(Aeq,1),2*n);...
        -M,eye(n),-eye(n)]);
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


%Define f such that sum over wp + wn
f=sparse([zeros(n,1);ones(2*n,1)]);

beq_full=[beq;-c];

%Set condition wp >= 0  and  wn >= 0
lb_full=sparse([lb;zeros(2*n,1)]);
ub_full=[ub;inf(2*n,1)];

x_help=linprog(f,A_full,b,Aeq_full,beq_full,lb_full,ub_full,options);

x=x_help(1:n);


end