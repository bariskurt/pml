function [gu] = mcp_update(g, x)
% CMPE58N_MCP_UPDATE Updates several gamma potentials with a single observation
%
% [gu] = cmpe58n_mcp_update(g, x)
%
% Inputs : 
% g : A collection of gamma potentials 
% x : Observation 
% 
% Outputs : 
% gu : Updated potentials 
% Usage Example : [] = cmpe58n_mcp_update();
%

% Change History : 
% Date Time Prog Note 
% 24-Nov-2009 2:46 PM ATC Created under MATLAB 7.7.0

% ATC = Ali Taylan Cemgil, 
% Department of Computer Engineering, Bogazici University
% e-mail : taylan.cemgil@boun.edu.tr


M = size(g, 1);
gu = zeros(size(g));

a = g(:,:, 1); 
b = g(:,:, 2); 
c = g(:,:, 3) + gammaln(a + x) - gammaln(a) - gammaln(x+1) + a.*log(b) - (a + x).*log(b+1);

gu = cat(3, a+ x, b+ 1, c);