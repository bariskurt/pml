% CMPE58N_MCP_POISS_INFERENCE Exact inference for the Poisson changepoint model
%
% Change History :
% Date Time Prog Note
% 26-Nov-2009 11:32 PM ATC Created under MATLAB 7.7.0
% ATC = Ali Taylan Cemgil,
% Department of Computer Engineering, Bogazici University
% e-mail : taylan.cemgil@boun.edu.tr

M = data.M;
% Potential
%  1st index :  mixture component
%  2nd        : time slice
%  3rd        : params of the gamma potential
eng.ff = zeros(M+1, M, 3);
eng.fp = zeros(M+1, M, 3);

% Backward potentials
eng.bf = zeros(M+1, M, 3);

for t=1:M
    if t==1,
        tau = 0;
        eng.fp(tau +1, t, :) = reshape([data.nu, data.B, data.log_p1], [1 1 3]);
        tau = 1;
        eng.fp(tau +1, t, :) = reshape([data.a0, data.b0, data.log_p0], [1 1 3]);
    else
        c = log_sum_exp(eng.ff((0:t-1) +1, t-1, 3), 1);
        tau = 0;
        eng.fp(0  +1, t, :) = reshape([data.nu, data.B, data.log_p1 + c], [1 1 3]);
        tau = 1:t;
        eng.fp(tau +1, t, :) = eng.ff(tau-1 +1, t-1, :);
        eng.fp(tau +1, t, 3) = eng.fp(tau +1, t, 3) + data.log_p0;
    end;
    
    % Filter
    eng.ff((0:t) +1, t, :) =  cmpe58n_mcp_update( eng.fp((0:t) +1, t, :), data.x(t));
end;

% Bacward pass
for t=M:-1:1,
    if t==M,
        tau = 0;
        eng.bf(tau +1, t, :) = reshape([data.x(t)+1, 1, 0], [1 1 3]);
    else        
        disp(eng.bf((0:(M-(t+1))) +1, t+1, :))        
        disp('-------------')
        temp = cmpe58n_mult_gampot(eng.bf((0:(M-(t+1))) +1, t+1, :), reshape([data.nu, data.B, 0], [1 1 3]));
        c = log_sum_exp(temp(:,1,3));
        disp(temp)
        disp('-----------------------------------')
        tau = 0;
        eng.bf(tau +1, t, :) = reshape([data.x(t)+1, 1, data.log_p1 + c], [1 1 3]);
        tau = 1:(M-t);
        eng.bf(tau +1, t, :) = cmpe58n_mult_gampot(eng.bf(tau-1 +1, t+1, :), reshape([data.x(t)+1, 1, data.log_p0], [1 1 3]) );
    end
end