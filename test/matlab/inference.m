% CMPE58N_MCP_POISS_INFERENCE Exact inference for the Poisson changepoint model
%

% Change History : 
% Date Time Prog Note 
% 26-Nov-2009 11:32 PM ATC Created under MATLAB 7.7.0

% ATC = Ali Taylan Cemgil, 
% Department of Computer Engineering, Bogazici University 
% e-mail : taylan.cemgil@boun.edu.tr
function [eng] = inference(data)

  M = data.M; 
  % Potential

  % 1st index : mixture component
  % 2nd : time slice
  % 3rd : params of the gamma potential

  eng.ff = zeros(M+1, M, 3); 
  eng.fp = zeros(M+1, M, 3);
   
  % Backward potentials 
  eng.bf = zeros(M+1, M, 3);
  
  % Filtered and smoothed cpp
  eng.cppf = zeros(2,M);
  eng.cpps = zeros(2,M);
  
  % Smoothed Mean
  eng.means = zeros(1,M);
  
  %data.log_p1 = log(0.80); 
  %data.log_p0 = log(1 - exp(data.log_p1));


  for t=1:M 
      if t==1,
          tau = 0; 
          eng.fp(tau +1, t, :) = reshape([data.a, data.b, data.log_p1], [1 1 3]); 
          tau = 1; 
          eng.fp(tau +1, t, :) = reshape([data.a, data.b, data.log_p0], [1 1 3]);
      else
          c = log_sum_exp(eng.ff((0:t-1) +1, t-1, 3), 1); 
          tau = 0;
          eng.fp(0 +1, t, :) = reshape([data.a, data.b, data.log_p1 + c], [1 1 3]); 
          tau = 1:t; 
          eng.fp(tau +1, t, :) = eng.ff(tau-1 +1, t-1, :);
          eng.fp(tau +1, t, 3) = eng.fp(tau +1, t, 3) + data.log_p0;
      end;
      
      % Filter
      eng.ff((0:t) +1, t, :) = mcp_update( eng.fp((0:t) +1, t, :), data.x(t));
  end;

  % Bacward pass
  for t=M:-1:1,
      if t==M, 
          tau = 0;
          eng.bf(tau +1, t, :) = reshape([data.x(t)+1, 1, 0], [1 1 3]);
      else
          tau = 0; 
          temp = mult_gampot(eng.bf((0:(M-(t+1))) +1, t+1, :), reshape([data.a, data.b, 0], [1 1 3])); 
          c = log_sum_exp(temp(:,1,3)); 
          eng.bf(tau +1, t, :) = reshape([data.x(t)+1, 1, data.log_p1 + c], [1 1 3]);

          tau = 1:(M-t);
          eng.bf(tau +1, t, :) = mult_gampot(eng.bf(tau-1 +1, t+1, :), reshape([data.x(t)+1, 1, data.log_p0],[1,1,3]));
      end
  end
  
  % forward filter change probabilities
  for j=1:M
      eng.cppf(1, j) = eng.ff(0 +1, j, 3); 
      eng.cppf(2, j) = log_sum_exp(eng.ff((1:j) +1, j, 3), 1);
  end;
  eng.cppf = normalize_exp(eng.cppf, 1);
  for j=M:-1:1,
      gamma = mult_gampot(eng.fp((0:j) +1, j, :), eng.bf( (0:(M-j)) +1, j, :) );
      [mean, cpp] = evaluate(gamma);
      eng.means(j) = mean;
      eng.cpps(:,j) = cpp;
      % eng.cpps(1,j) = cpp;
      % c = gamma(2:end, :, 3); 
      % eng.cpps(2,j) = log_sum_exp(c(:), 1);
      %{
      eng.cpps(1,j) = log_sum_exp(gamma(1, :, 3),2); 
      c = gamma(2:end, :, 3); 
      eng.cpps(2,j) = log_sum_exp(c(:), 1);
      %}
      
  end
  % eng.cpps = normalize_exp(eng.cpps, 1);
  
  % Mean and cpp
  %{
  eng.mean = zeros(1, M);
  eng.cpp = zeros(1, M);
  for j=M:-1:1,
    gamma = mult_gampot(eng.fp((0:j) +1, j, :), eng.bf( (0:(M-j)) +1, j, :) );
    [eng.mean(j), eng.cpp(j)] = evaluate(gamma)
  end
  %}
  

end % function