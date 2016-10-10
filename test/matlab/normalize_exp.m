function [r] = normalize_exp(X, dim)
% NORMALIZE_EXP		Numerically Stable normalize(exp(X), dim)
%
%  [] = normalize_exp()
%
% Inputs :
%	:
%
% Outputs :
%	:
%
% Usage Example : [] = normalize_exp();
%
%
% Note	:
% See also

% Uses :

% Change History :
% Date		Time		Prog	Note
% 03-Jun-2002	 3:17 PM	ATC	Created under MATLAB 5.3.1.29215a (R11.1)

% ATC = Ali Taylan Cemgil,
% SNN - University of Nijmegen, Department of Medical Physics and Biophysics
% e-mail : cemgil@mbfys.kun.nl 

  if ~exist('dim', 'var'), dim = 1; end;
  
  mx = max(X, [], dim);
  sz = size(X);
  sz2 = ones(size(sz));
  sz2(dim) = sz(dim);
  X2 = X - repmat(mx, sz2);
  
  r = normalize(exp(X2), dim);