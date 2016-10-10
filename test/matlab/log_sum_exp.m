function L=log_sum_exp(X,dim)
%LOG_SUM_EXP  Calculates log(sum(exp(X)) operation safely.
%
%   L=log_sum_exp(X,dim) Sums the exponentials of elements in the X matrix
%   by avoiding round off error, along the dimension dim. Again the
%   logarithm of the result is returned.

% Change History :
% Date          Programmer 
% ?             Ali Taylan Cemgil.

if ~exist('dim', 'var'), dim = 1; end;
sz = size(X);
sz2 = ones(size(sz));
sz2(dim) = sz(dim);

mx = max(X, [], dim);
mx2 = repmat(mx, sz2);

L = mx + log(sum(exp(X-mx2),dim));
    
end