function [gp] = mult_gampot(g1, g2)
% CMPE58N_MULT_GAMPOT Multiplies gamma potentials 
% 
% Inputs : 
% g1, g2 gamma potentials 
% 
% Outputs : 
% gp : Coefficients of exp(c_1)Gamma(x; a_1, b_1)exp(c_2)Gamma(x; a_2, b_2) 
%

% Change History : 
% Date Time Prog Note 
% 24-Nov-2009 2:19 PM ATC Created under MATLAB 7.7.0

% ATC = Ali Taylan Cemgil, 
% Department of Computer Engineering, Bogazici University 
% e-mail : taylan.cemgil@boun.edu.tr

M1 = size(g1,1); 
M2 = size(g2,1); 
gp = zeros(M1, M2, 3);

%a = a_1 + a_2 - 1;
gp(:,:,1) = repmat(g1(:,1,1), [1 M2]) + repmat(g2(:,1,1)', [M1 1]) - 1;

%b = b_1 + b_2;
gp(:,:,2) = repmat(g1(:,1,2), [1 M2]) + repmat(g2(:,1,2)', [M1 1]);

%c = c_1 + c_2 + gammaln(a_1 + a_2 - 1) - (a_1 + a_2 - 1).*log(b_1 + b_2) 
%    - gammaln(a_1) - gammaln(a_2) + a_1.*log(b_1) + a_2.*log(b_2)

a_1 = repmat(g1(:,1,1), [1 M2]);
a_2 = repmat(g2(:,1,1)', [M1 1]);

b_1 = repmat(g1(:,1,2), [1 M2]);
b_2 = repmat(g2(:,1,2)', [M1 1]);

gp(:,:,3) = repmat(g1(:,1,3), [1 M2]) + repmat(g2(:,1,3)', [M1 1]) ...
            + gammaln(gp(:,:,1)) - gp(:,:,1).*log(gp(:,:,2)) ...
            - gammaln(a_1) - gammaln(a_2) + a_1.*log(b_1) + a_2.*log(b_2); 
            