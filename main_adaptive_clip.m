% This is the main file for running the experiments in the paper 
% 
% Sinan Yildirim (2023), "Differentially Private Online Bayesian Estimation
% With Adaptive Truncation"
%
% Sinan Yildirim, 
% 19.01.2023

clear; close all; clc; fc = 0;

rng(1);
% true parameter
mu_x = 50; var_x = 10; std_x = sqrt(var_x); theta_true = [mu_x std_x]';
eps_DP = 1; % privacy parameter
n = 1000; % data size

% iteration and particle numbers and window length
M = 30; % number of experiments
N = 1000; % number of SMC particles
M_MCMC = 100000; % number of iterations for MCMC
N_MHAAR = 20; % number of particles in MHAAR

% Score function for the FIM
symmetric_int = 1;
if symmetric_int == 1
    score_fn = @(F) F(1, 1);
elseif symmetric_int == 0
    score_fn = @(F) F(1, 1)+F(2, 2);
end
% Determine the intervals
n_FIM = 1000; N_FIM = 10000; R_FIM = 51;
fprintf('Calculating the best interval... \n');
[A_best, B_best, F_AB] = find_best_a_b_norm(eps_DP, [-3 3], n_FIM, ...
    N_FIM, R_FIM, score_fn, symmetric_int);

% Initialisation
alg_names = {'SMC adapt.', 'SMC non-adapt wide int.', 'MCMC wide int'};
Mu_Par = cell(2, M); Std_Par = cell(2, M);
L_SMC = cell(2, M); R_SMC = cell(2, M);
Theta_samp_MHAAR = cell(1, M);
Theta_est = repmat({zeros(2, M)}, 1, 3);

f_int = 10;
L_f = mu_x - f_int*std_x; R_f = mu_x + f_int*std_x;

% Prior and proposal parameters
prior_params.mu_prior_m = 0;
prior_params.mu_prior_v = 10000;
prior_params.var_prior_alpha = 1;
prior_params.var_prior_beta = 1;
prop_params.sigma_q = 3;
prop_params.sigma_q_mu = 10/sqrt(n);
prop_params.sigma_q_std = 20/sqrt(n);

% This is the vector of time steps 
t_plot = 20:20:n;

%% Experiments
for i = 1:M
    disp(i);
    % Generate the true data
    x = mu_x + randn(1, n)*sqrt(var_x);

    % 1. SMC estimation with adaptive intervals
    fprintf('%s is running...\n', alg_names{1});
    [Mu_par, Std_par, ~, l_SMC, r_SMC] = SMC_DP_norm(x, eps_DP, ...
        [A_best, B_best], A_best, B_best, N, prior_params, prop_params, 1);
    Theta_est{1}(:, i) = [mean(Mu_par(:, end)), mean(Std_par(:, end))]';
    Mu_Par{1, i} = Mu_par(:, t_plot);
    Std_Par{1, i} = Std_par(:, t_plot);
    L_SMC{1, i} = l_SMC(t_plot);
    R_SMC{1, i} = r_SMC(t_plot);
    clear Mu_par Std_par l_SMC r_SMC;

    % 2. SMC estimation with non-adaptive intervals around the mean
    fprintf('%s is running...\n', alg_names{2});
    [Mu_par, Std_par, ~, l_SMC, r_SMC] = SMC_DP_norm(x, eps_DP, ...
        [A_best, B_best], L_f, R_f, N, prior_params, prop_params, 0);
    Theta_est{2}(:, i) = [mean(Mu_par(:, end)), mean(Std_par(:, end))]';
    Mu_Par{2, i} = Mu_par(:, t_plot);
    Std_Par{2, i} = Std_par(:, t_plot);
    L_SMC{2, i} = l_SMC(t_plot);
    R_SMC{2, i} = r_SMC(t_plot);
    clear Mu_par Std_par l_SMC r_SMC;

    % 3. MCMC algorithm without adaptive choice of truncation
    fprintf('%s is running...\n', alg_names{3});
    y_f = trunc_lr(x, L_f, R_f) + laprnd(1, n, 0, (R_f-L_f)/eps_DP);
    [outputs] = MHAAR_DP_norm(y_f, L_f, R_f, eps_DP, M_MCMC, N_MHAAR,...
        prior_params, prop_params);
    Theta_est{3}(:, i) = mean(outputs.Thetas(:, (4*end/5+1):end), 2);
    Theta_samp_MHAAR{1, i} = outputs.Thetas;
    clear outputs;
end

filename = [sprintf('symm_%d_n_%d_M_%d_eps_DP_%02d_f_%d',...
    symmetric_int, n, M, eps_DP*100, f_int) '_' date];
save(filename);


%% Plot the results

%%% A. Scatter plots for the particles of SMC algorithms
fc = fc + 1; figure(fc);
L_t = length(t_plot);
m_plot = 1;
n_alg = 2;
for i = 1:n_alg
    subplot(1, n_alg*2, i + 2*(n_alg-2));
    plot_data = Mu_Par{i, m_plot};
    plot(reshape(repmat(t_plot, N, 1), L_t*N, 1), plot_data(:), '.')
    xlabel('$t$', 'Interpreter', 'Latex');
    ylabel('particle values', 'Interpreter', 'Latex');
    hold on;
    plot(t_plot, mean(Mu_Par{i, m_plot}, 1), 'r');
    set(gca, 'ylim', [10, 90]);
    plot(t_plot, L_SMC{i, m_plot}, 'k');
    plot(t_plot, R_SMC{i, m_plot}, 'k');
    plot(1:n, mu_x*ones(1, n), 'k');
    title(['$\mu$ -', sprintf('%s', alg_names{i})], 'Interpreter', 'Latex');
    hold off;

    subplot(1, n_alg*2, i + 2*(n_alg-1));
    plot_data = Std_Par{i, m_plot};
    plot(reshape(repmat(t_plot, N, 1), L_t*N, 1), log(plot_data(:)), '.');
    hold on;
    plot(t_plot, log(mean(Std_Par{i, m_plot}, 1)), 'r');
    plot(1:n, log(std_x)*ones(1, n), 'k');
    xlabel('$t$', 'Interpreter', 'Latex');
    ylabel('(log-)particle values', 'Interpreter', 'Latex');
    title(['$\sigma$ - ' sprintf('%s', alg_names{i})], 'Interpreter', 'Latex');
    hold off;
end

%%% B. Box plots for all the algorithms
n_alg = 3;
Mu_est = zeros(M, 3);
Std_est = zeros(M, 3);
for i = 1:3
    Mu_est(:, i) = Theta_est{i}(1, :)';
    Std_est(:, i) = Theta_est{i}(2, :)';
end

fc = fc + 1; figure(fc);
subplot(1, 2, 1);
boxplot(Mu_est);
set(gca, 'XTickLabel', {'SMC adpt', 'SMC non-adpt', 'MCMC'});
hold on;
plot(0:(n_alg+1), mu_x*ones(1, n_alg+2), '-.k');
hold off;
title('$\mu$', 'Interpreter', 'Latex');
set(gca, 'ylim', [45, 55]);
subplot(1, 2, 2);
boxplot(Std_est);
set(gca, 'XTickLabel', {'SMC adpt', 'SMC non-adpt', 'MCMC'});
hold on;
plot(0:(n_alg+1), std_x*ones(1, n_alg+2), '-.k');
hold off;
set(gca, 'ylim', [0.5, 4]);
title('$\sigma$', 'Interpreter', 'Latex');