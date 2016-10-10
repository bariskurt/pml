function [A, Z] = normalize(A, dim)
% NORMALIZE		Normalize a Matrix along a dimension
%
%  [A, Z] = NORMALIZE(A, <DIM>)
%
% Inputs :
%      A : A Matrix
%      DIM : Dimension to normalize over <Default = empty>
%
% Outputs :
%      A : A Matrix where the sum over dim gives all ones
%      Z : Normalization constant. 
%
% Usage Example : [A, Z] = normalize([1 2 3]);
%
%
% Note	:
% See also

% Uses :

% Change History :
% Date		Time		Prog	Note
% 21-Sep-2000	 1:20 PM	ATC	Removed a bug
% 09-Nov-1999	11:27 AM	ATC	Created under MATLAB 5.2.0.3084

% ATC = Ali Taylan Cemgil,
% SNN - University of Nijmegen, Department of Medical Physics and Biophysics
% e-mail : cemgil@mbfys.kun.nl 

if nargin<2, 
  Z = sum(A(:));

  % Zeros to one before division
  Z = Z + (Z==0);
  A = A / Z;
else
  Z = sum(A, dim);
  Z = Z + (Z==0);
  dm = ones(1, length(size(A)));
  dm(dim) = size(A, dim); 
  A = A./repmat(Z, dm);
end;
