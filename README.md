# SMC_DP_adaTr
This repository contains the code that replicates the experiments in the arXiv paper

Sinan Yildirim (2023), "Differentially Private Online Bayesian Estimation With Adaptive Truncation"

To repeat the experiments that result in Figure 2 and 3 of the paper, run the main script named
"main_adaptive_clip.m" 
separately with epsilon = 1, 2, 5, 10.

The functions that correspond to Bayesian estimation algorithms and the procedure to find the best truncation points are 
1) The SMC algorithm with (optional) adaptive truncation

[Mu_par, Std_par, X_par, L, R] = SMC_DP_norm(X_true, eps_DP, AB,...
    L0, R0, N, prior_params, prop_params, update_lr)
    
where
- X_true is the sensitive data
- eps_DP is the epsilon value of DP
- AB is a 2x1 vector has the best a, b values (will be used to determine the truncation points)
- L0, R0 are the left and right points of the initial truncation interval
- N is the number of particles
- prior_params is a struct that contains the hyperparameters for the prior distributions of mu and var_x of the normal distribution
- prop_params is a struct that contains the proposal parameters for the MCMC moves within the SMC algorithm that updates the X_{t} components of the particles
- update_lr is a binary input, taking 1 if adaptive truncation is ON and 0 otherwise (in which case the interval [L0 R0] is used throughout.
  
- Mu_par is a matrix that contains the particle values for mu_x at all time steps
- Std_par is a matrix that contains the particle values for std_x at all time steps
- X_par is a matrix that contains the particle values for X_t's at the last time step
- L and R are vectors for the left and right truncation points determined by the algorithm at all time steps

2) MHAAR algorithm for batch Bayesian estimation

[outputs] = MHAAR_DP_norm(Y, L, R, eps_DP, M, N, prior_params, prop_params)

where
- Y are the observations,
- L, R are vectors for the left and right truncation points at all time steps
- eps_DP is the epsilon value of DP 
- M is the number of iterations
- N is the sampled size used for the auxiliary variable of the joint distribution targeted by MHAAR
- prior_params is a struct that contains the hyperparameters for the prior distributions of mu and var_x of the normal distribution
- prop_params is a struct that contains the proposal parameters for the parameters

3) The function for finding the best a and b values for the normal distribution

[A_max, B_max, FIM_all] = find_best_a_b_norm(eps_DP, AB_lims,...
    n, N, R, score_fn, symmetric_int)
    
where
- eps_DP is the epsilon value of DP 
- AB_lims is the left and right limits of the grid of values for a, b where the FIM is calculated
- n is the sample size for the observations
- N is the sample size for self-normalised importance sampling that is used to calculate the score vector for a given observation
- R is the resolution of the grid of (a, b) values ranging according to AB_limits (the grid is R x R)
- score_fn is a function handle and stands for the score function for the fisher information matrix
- symmetric_int is a binary variable indicating whether the best (a, b) must satisfy b = -a. If so, then only the `diagonals' of the grid are searched for
 computational saving.
 
For questions: sinanyildirim@sabanciuniv.edu
