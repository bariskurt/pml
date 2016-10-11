function [mean, cpp] = evaluate (message)
  % cpp calculation
  tmp = [0,0];
  tmp(1) = log_sum_exp(message(1, :, 3),2);
  c = message(2:end, :, 3); 
  tmp(2) = log_sum_exp(c(:), 1);
  cpp = normalize_exp(tmp,2);
  % mean calculation
  consts = message(:, :, 3);
  norm_consts = normalize_exp(consts(:),1);
  norm_consts = reshape(norm_consts,size(message,1),size(message,2));
  mean = sum(sum(norm_consts.*message(:,:,1)./message(:,:,2)));
  %{
  mean = 0;
  for i=1:size(message,1)
     for j=1:size(message,2)
         mean = mean + norm_consts(i,j)*message(i,j,1)/message(i,j,2)
     end
  end
  %}
end
