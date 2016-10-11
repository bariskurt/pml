function model = learn(data)

    M = data.M;
    MIN_ITER = 10;
    MAX_ITER = 2;
    ll = zeros(MAX_ITER);
    
    for iter = 1:MAX_ITER,
        
        eng = inference(data);
        
        % likelihood:
        ll(iter) = log_sum_exp(eng.ff(:, M, 3)); 
        fprintf(1,'ll is %.12f\n', ll(iter));
        
        if (iter > 1) &&  ll(iter) < ll(iter-1),
            fprintf(1, 'Likelihood decreased : %f\n', ll(iter-1) - ll(iter) );
        end

        % E-Step
        cpp = zeros(M,2); 
        ss = zeros(M,2);        
        for j=M:-1:1,
            gamma = mult_gampot(eng.fp((0:j) +1, j, :), eng.bf( (0:(M-j)) +1, j, :) );
            [~, cpp(j,:)] = evaluate(gamma);
            ss(j,:) = compute_ss(gamma);
        end
        cpp_sum = sum(cpp(:,1));
        ss = sum(ss .* repmat(cpp(:,1),1,2), 1) / cpp_sum;
        
        % M-Step
        data.p1 = cpp_sum / M;
        [data.a, data.b] = gamma_est(ss);
        
        fprintf(1,'a = %.12f, b = %.12f\n', data.a, 1/data.b);
        fprintf(1,'p1 = %.12f\n', data.p1);
        fprintf(1,'--------------\n');
        
    end
    
    eng = inference(data);
    
end