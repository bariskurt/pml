function [lg] = cmpe58n_eval_mogamma(g, lambda)
% CMPE58N_EVAL_MOGAMMA Evaluates a mixture of gamma densities
%
% Inputs :
% g : N x 1 x 3 array of gamma potential parameters
%          lambda : grid to evaluate the posterior
%
% Outputs :
% lg : log density
%
% Change History :
% Date Time Prog Note
% 24-Nov-2009  4:33 PM ATC Created under MATLAB 7.7.0
% ATC = Ali Taylan Cemgil,
% Department of Computer Engineering, Bogazici University
% e-mail : taylan.cemgil@boun.edu.tr
J = size(g, 1);
lg = zeros(size(lambda));
Z =  log_sum_exp(g(:, 1, 3),1);
for j=1:J,
    a = g(j, 1, 1);
    b = g(j, 1, 2);
    lg = lg + exp(g(j,1,3) - Z + (a - 1).*log(lambda) - b.*lambda - gammaln(a) + a.*log(b));
end;