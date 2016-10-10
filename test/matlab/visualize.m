% CMPE58N_MCP_POISS_VISUALISE Compute smoothed estimates and plot

% Change History : 
% Date Time Prog Note 
% 26-Nov-2009 11:33 PM ATC Created under MATLAB 7.7.0
% ATC = Ali Taylan Cemgil, 
% Department of Computer Engineering, Bogazici University 
% e-mail : taylan.cemgil@boun.edu.tr

lam = linspace(0.01,max(data.x),100);

eng.pr = zeros(2, M); 
eng.prs = zeros(2, M);

LAM = zeros(length(lam), M); 
LAMS = zeros(length(lam), M);

%Evaluate filtered estimates of indicators 
for j=1:M
    eng.pr(1, j) = eng.ff(0 +1, j, 3); 
    eng.pr(2, j) = log_sum_exp(eng.ff((1:j) +1, j, 3), 1);
    LAM(:, j) = eval_mogamma(eng.ff((0:j) +1, j, :), lam); 
end;


%Evaluate smoothed densities 
for j=M:-1:1,
    gamma = mult_gampot(eng.fp((0:j) +1, j, :), eng.bf( (0:(M-j)) +1, j, :) );
    eng.prs(1,j) = log_sum_exp(gamma(1, :, 3),2); 
    c = gamma(2:end, :, 3); 
    eng.prs(2,j) = log_sum_exp(c(:), 1);
    LAMS(:, j) = eval_mogamma(reshape(gamma, [size(gamma,1)*size(gamma,2) 1 3]), lam); 
end

subplot(411) 
stem(data.x) 
hold on 
plot((data.lambda), 'r.') 
%ln= grid_line(find(data.r)); 
%set(ln, 'lines', ':') 
hold off

subplot(412) 
imagesc(1:M, lam, LAM); 
set(gca, 'ydir', 'n') 
ylabel('filtered intensity')

subplot(413); 
pr = normalize_exp(eng.pr, 1); 
plot(pr(1,:)) 
prs = normalize_exp(eng.prs, 1); 
plot(prs(1,:)) 
ylabel('smoothed cp probability')

subplot(414) 
imagesc( 1:M, (lam), LAMS); 
set(gca, 'ydir', 'n') 
ylabel('smoothed intensity')
colormap(flipud(gray))