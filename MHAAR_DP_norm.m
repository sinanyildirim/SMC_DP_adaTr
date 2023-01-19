function [outputs] = MHAAR_DP_norm(Y, L, R, eps_DP, M, N, prior_params, prop_params)

% MHAAR_DP_norm(Y, L, R, eps_DP, M, N, prior_params, prop_params)
% 
% This function implements MHAAR algorithm to infer the parameters mu and
% var_x from the observations
% 
% Y_t = min(max(L(t), X_t), R(t)) + Laplace(DP_param), X_t ~ N(mu_x, var_x), t =
% 1,..n
%
% The theoretical details for this MHAAR algorithm can be found in Section
% 4 of 
% Andrieu, C., Yıldırım, S., Doucet, A., and Chopin, N. (2020). 
% Metropolis-Hastings with averaged acceptance ratios. arXiv:2101.01253
%
% Sinan Yildirim
% 19.01.2023

sigma_q_mu = prop_params.sigma_q_mu;
sigma_q_std = prop_params.sigma_q_std;

alpha_ast_mu = 0.1;
alpha_ast_std = 0.1;
S_acc_mu = 0;
S_acc_std = 0;

mu_prior_m = prior_params.mu_prior_m;
mu_prior_v = prior_params.mu_prior_v;
var_prior_alpha = prior_params.var_prior_alpha;
var_prior_beta = prior_params.var_prior_beta;

n = length(Y);

DP_param = (R-L)/eps_DP;

theta = [mean(Y) sqrt(max(0.01, var(Y) - DP_param^2))]';
Thetas = zeros(2, M);

z = randn(1, n);

update_a_burn_in = M/10;
sigma_q_mu_vec = zeros(1, M);
sigma_q_std_vec = zeros(1, M);


for m = 1:M

    % Rejuvenate using MCMC
    % propose new values
    theta_prop(1) = theta(1) + sigma_q_mu*randn;
    temp_prop = theta(2)^2 + sigma_q_std*randn;
    theta_prop(2) = sqrt(temp_prop);
    
    % sample particles
    Z = [z; randn(N-1, n)];

    % calculate the weights
    X_par_0 = Z*theta(2) + theta(1);
    S_par_0 = trunc_lr(X_par_0, L, R);
    log_w_0 = -abs(Y - S_par_0)/DP_param;
    w_max_0 = max(log_w_0, [], 1);
    log_sum_0 = log(sum(exp(log_w_0 - w_max_0), 1)) + w_max_0;
  
    if temp_prop > 0  
        X_par_1 = Z*theta_prop(2) + theta_prop(1);
        S_par_1 = trunc_lr(X_par_1, L, R);    
        log_w_1 = -abs(Y - S_par_1)/DP_param;
        w_max_1 = max(log_w_1, [], 1);
        log_sum_1 = log(sum(exp(log_w_1 - w_max_1), 1)) + w_max_1;
        
        % calculate the acceptance ratio
        log_prior_ratio = -0.5*((theta_prop(1) - mu_prior_m)^2/mu_prior_v...
            - (theta(1) - mu_prior_m)^2/mu_prior_v)...
            - (var_prior_alpha + 1)*2*(log(theta_prop(2)) - log(theta(2))) ...
            - var_prior_beta*(1/theta_prop(2)^2 - 1/theta(2)^2);
        
        log_r = sum(log_sum_1 - log_sum_0) + log_prior_ratio;
    
        % decision and sampling of z
        decision = rand < exp(log_r);
    else
        decision = 0;
    end

    if decision == 1
        theta = theta_prop;
        W_temp = exp(log_w_1 - log_sum_1);
        temp_ind_vec = sum(rand(1, n) > cumsum(W_temp)) + 1;
        temp_mtx_ind_vec = N*(0:n-1) + temp_ind_vec;
        z = Z(temp_mtx_ind_vec);
    else
        W_temp = exp(log_sum_0 - log_sum_0);
        temp_ind_vec = sum(rand(1, n) > cumsum(W_temp)) + 1;
        temp_mtx_ind_vec = N*(0:n-1) + temp_ind_vec;
        z = Z(temp_mtx_ind_vec);
    end
   
    % store the sample
    Thetas(:, m) = theta;

    % update the proposal variances
    gamma_s = m^(-0.6);
    gamma_q = m^(-0.6);

    if mod(m, 2)==0
        S_acc_mu = (1 - gamma_s)*S_acc_mu + gamma_s*decision;
        if m > update_a_burn_in
            sigma_q_mu = sigma_q_mu + gamma_q*(S_acc_mu - alpha_ast_mu);
        end
    else
        S_acc_std = (1 - gamma_s)*S_acc_std + gamma_s*decision;
        if m > update_a_burn_in
            sigma_q_std = sigma_q_std + gamma_q*(S_acc_std - alpha_ast_std);
        end
    end

    sigma_q_mu_vec(m) = sigma_q_mu;
    sigma_q_std_vec(m) = sigma_q_std;
    
end

outputs.Thetas = Thetas;
outputs.sigma_q_mu_vec = sigma_q_mu_vec;
outputs.sigma_q_std_vec = sigma_q_std_vec;