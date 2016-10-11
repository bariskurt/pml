 % B.1 Data generation

% CMPE58N_MCP_POISS Script to generate data from the model 
%

% Change History : 
% Date Time Prog Note 
% 01-Apr-2009 9:40 AM ATC Created under MATLAB 7.7.0

% ATC = Ali Taylan Cemgil,
% Department of Computer Engineering, Bogazici University 
% e-mail : taylan.cemgil@boun.edu.tr

function [data] = gen_data()

  % Set the Hyperparameters 
  data.M = 90; 
  data.a = 10; 
  data.b = 1; 
  data.p1 = 0.2;
  data.log_p1 = log(data.p1); 
  data.log_p0 = log(1 - data.p1);

  data.r = double(rand(1, data.M) < exp(data.log_p1));   
  data.lambda = zeros(1, data.M); 
  data.lambda(1) = gamrnd(data.a, data.b);
  

  for t=2:data.M,
      if data.r(t)==1,
          data.lambda(t) = gamrnd(data.a, data.b);
      else 
          data.lambda(t) = data.lambda(t-1);                   
      end;
  end;

  data.x = poissrnd(data.lambda);

  % Plot data 
  close all;
  stem(data.x);
  hold on;
  plot(1:data.M, [data.lambda], 'r.');
  hold off;
  legend('x','\lambda'); xlabel('t'); ylabel('x_t');
  
  % Save Data
  saveTxt('/tmp/states.txt', data.lambda);
  saveTxt('/tmp/obs.txt', data.x);
  
end