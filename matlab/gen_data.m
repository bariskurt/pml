% CMPE58N_MCP_POISS Script to generate data from the model
%
% Change History :
% Date Time Prog Note
% 01-Apr-2009  9:40 AM ATC Created under MATLAB 7.7.0
% ATC = Ali Taylan Cemgil,
% Department of Computer Engineering, Bogazici University
% e-mail : taylan.cemgil@boun.edu.tr
% Set the Hyperparameters

M = 100;
data.M = M;
data.nu = 0.9;  
data.B = 0.1;
data.a0 = 5; 
data.b0 = 0.2;
data.log_p1 = log(0.05); 
data.log_p0 = log(1 - exp(data.log_p1));
data.r = double(rand(1, M) < exp(data.log_p1));
data.lambda0 = gamrnd(data.a0, 1/data.b0);
data.lambda = zeros(1, M);
for t=1:M,
    if data.r(t)==1,
        data.lambda(t) =  gamrnd(data.nu, 1/data.B);
    else
        if t>1,
            data.lambda(t) = data.lambda(t-1);
        else
            data.lambda(t) = data.lambda0;
        end;
    end;
end;
data.x = poissrnd(data.lambda);

% Plot data
stem(data.x)
hold on
plot(0:M, [data.lambda0 data.lambda], 'r.')
%ln= grid_line(find(data.r)); set(ln, 'lines', ':')
%hold off
%legend('x','\lambda'); xlabel('t'); ylabel('x_t')