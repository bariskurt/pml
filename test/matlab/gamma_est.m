function [a, b] = gamma_est(ss)

    mean_x = ss(1);
    mean_log_x = ss(2);
    log_mean_x = log(mean_x);
    
    a = 0.5 / (log_mean_x - mean_log_x);        
    for i=1:5,
        temp = mean_log_x - log_mean_x + log(a) - psi(a);
        temp = temp / (a * a *(1/a - psi(1,a)));
        a = 1/(1/a + temp);        
    end    
    
    b = a / mean_x;
end