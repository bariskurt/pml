function ss = compute_ss(message)

  consts = message(:, :, 3);
  norm_consts = normalize_exp(consts(:),1);
  
  norm_consts = reshape(norm_consts,size(message,1),size(message,2));
  
  mean = sum(sum(norm_consts.*message(:,:,1)./message(:,:,2)));   
  
  logX = psi(message(:,:,1)) - log(message(:,:,2));  
  mean_log = sum(sum(norm_consts.*logX));
  
  ss = [mean, mean_log];
  
end