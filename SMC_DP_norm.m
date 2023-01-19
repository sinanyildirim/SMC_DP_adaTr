function [Mu_par, Std_par, X_par, L, R] = SMC_DP_norm(X_true, eps_DP, AB,...
    L0, R0, N, prior_params, prop_params, update_lr)

% [Mu_par, Std_par, X_par, L, R] = SMC_DP_norm(X_true, eps_DP, AB, L0, R0, 
% N, prior_params, prop_params, update_lr)
% 
% This function implements the SMC algorithm with adaptive truncation for
% differentially private online Bayesian estimation of the parameters of a
% normal distribution when the observations are
% 
% Y_t = min(max(L(t), X_t), R(t)) + (R(t) - L(t))*Laplace(1/eps_DP), 
% X_t ~ N(mu_x, var_x), t = 1,..n.
% 
% If update_lr == 1, the truncation points are chosen adaptively according
% to an exploration-exploitation mechanism a la Thompson sampling.
% 
% Sinan Yildirim
% 19.01.2023

% proposal parameter
sigma_q = prop_params.sigma_q;

% Prior parameters
mu_prior_m = prior_params.mu_prior_m;
mu_prior_v = prior_params.mu_prior_v;
var_prior_alpha = prior_params.var_prior_alpha;
var_prior_beta = prior_params.var_prior_beta;

n = length(X_true);
Y = zeros(1, n);
DP_param_vec = zeros(1, n);

A = AB(1); B = AB(2);
if update_lr == 0
    L = ones(1, n)*L0;
    R = ones(1, n)*R0;
else
    L = [L0 zeros(1, n-1)];
    R = [R0 zeros(1, n-1)];
end

Mu_par = zeros(N, n);
Std_par = zeros(N, n);

for t = 1:n
    % DP noise variance
    DP_param = (R(t) - L(t))/eps_DP;
    DP_param_vec(t) = DP_param;

    
    y =  trunc_lr(X_true(t), L(t), R(t)) + laprnd(1, 1, 0, DP_param);
    Y(t) = y;

    % propagate the particles
    if t == 1
        mu_par = y*ones(N, 1); 
        std_par = 1*ones(N, 1);
        X_par = mu_par;        
    else
        X_par = [X_par_prev mu_par + randn(N, 1).*std_par];
    end

    % weight the particles
    mean_y_par = trunc_lr(X_par(:, t), L(t), R(t));    
    log_w = -abs(y - mean_y_par)./DP_param;

    w_max = max(log_w, [], 1);
    log_sum_w = log(sum(exp(log_w - w_max), 1)) + w_max;
    w_par = exp(log_w - log_sum_w);

    % Resample particles
    % res_ind = randsample(1:N, N, 'true', w_par);
    res_ind = resample(w_par, 'systematic');

    X_par = X_par(res_ind, :);
    mu_par = mu_par(res_ind);
    std_par = std_par(res_ind);

    % Rejuvenate using MCMC
    % 1. propose new values with random walk proposal
    X_par_prop = X_par + sigma_q*std_par.*randn(N, t);

    % acceptance ratio    
    log_prior_prop = -(X_par_prop - mu_par).^2./(2*std_par.^2);
    log_prior_curr = -(X_par - mu_par).^2./(2*std_par.^2);
    
    log_p_prop = -abs(Y(1:t) - trunc_lr(X_par_prop, L(1:t), R(1:t)))./DP_param_vec(1:t);
    log_p_curr = -abs(Y(1:t) - trunc_lr(X_par, L(1:t), R(1:t)))./DP_param_vec(1:t);

    log_ar = log_p_prop - log_p_curr + log_prior_prop - log_prior_curr;

    decision_mtx = rand(N, t) < exp(log_ar);
    X_par(decision_mtx) = X_par_prop(decision_mtx);

    % 2. propose new values from the prior of X
    X_par_prop = mu_par + std_par.*randn(N, t);
    
    log_p_prop = -abs(Y(1:t) - trunc_lr(X_par_prop, L(1:t), R(1:t)))./DP_param_vec(1:t);
    log_p_curr = -abs(Y(1:t) - trunc_lr(X_par, L(1:t), R(1:t)))./DP_param_vec(1:t);

    log_ar = log_p_prop - log_p_curr;

    decision_mtx = rand(N, t) < exp(log_ar);
    X_par(decision_mtx) = X_par_prop(decision_mtx);

    % update theta
    v_mu_post = 1./(1/mu_prior_v + t./(std_par.^2));
    m_mu_post = v_mu_post.*(mu_prior_m/mu_prior_v + sum(X_par, 2)./(std_par.^2));

    mu_par = m_mu_post + sqrt(v_mu_post).*randn(N, 1);
    
    % variance
    alpha_post = var_prior_alpha + t/2;
    beta_post = var_prior_beta + 0.5*sum((X_par - mu_par).^2, 2);
    std_par = sqrt(1./gamrnd(alpha_post, 1./beta_post));
    
    % keep previous particles
    X_par_prev = X_par;

    % determine the truncation points
    if t < n && update_lr == 1
        res_ind = randsample(1:N, 1);
        theta_trunc = [mu_par(res_ind) std_par(res_ind)];
        L(t+1) = theta_trunc(2)*A + theta_trunc(1);
        R(t+1) = theta_trunc(2)*B + theta_trunc(1);
    end

    Mu_par(:, t) = mu_par;
    Std_par(:, t) = std_par;

end    