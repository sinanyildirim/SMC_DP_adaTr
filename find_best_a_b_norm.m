function [A_max, B_max, FIM_all] = find_best_a_b_norm(eps_DP, AB_lims,...
    n, N, R, score_fn, symmetric_int)

% [A_max, B_max, FIM_all] = find_best_a_b_norm(eps_DP, AB_lims,...
%   n, N, R, score_fn, symmetric_int)
%
% This function finds the best truncation pair (a, b) according to the FIM
% of truncated and cluttered observations
%
% Y = max(a, min(b, X)) + (b-a)*Laplace(0, 1/eps_DP), X ~ N(0, 1)
%
% The FIM is calculated numerically using Monte Carlo. The approximation is
% based on the expectation of outer products of the score vectors.
%
% Sinan Yildirim

sc_FIM_max = 0;

x_temp = randn(1, n);
v0_temp = laprnd(1, n, 0, 1);

% If symmetric intervals are required, check only intervals of form [-A, A] 
if symmetric_int == 0
    AB_vec = linspace(AB_lims(1), AB_lims(2), R);
    [range_A, range_B] = ndgrid(AB_vec, AB_vec);
    range_AB_temp = [range_B(:) range_A(:)];
    range_AB = zeros(R*(R-1)/2, 2);
    j = 0;
    for i = 1:R
        range_AB((j+1):(j + R-i), :) = range_AB_temp(((i-1)*R+i+1): (i*R), :);
        j = j + R-i;
        
    end
    plot(range_A);
%    plot(range_AB(:, 1), range_AB(:, 2));
else
    range_A = linspace(AB_lims(1), 0, R);
    range_AB = [range_A(:) (-1)*range_A(:)];
end

nR = length(range_AB);
Sc = zeros(1, nR);
FIM_all = zeros(4, nR);

for i = 1:nR
    if mod(i, 100) == 0
        disp(i);
    end

    a = range_AB(i, 1);
    b = range_AB(i, 2);
    
    DP_param = (b-a)/eps_DP;

    % Generate observations
    y = trunc_lr(x_temp, a, b) + DP_param*v0_temp;

    % This is a 2 x n matrix
    [~, ~, Grad_y_vec] = score_est_norm(y, [0, 1], DP_param, a, b, N);

    % averaging of the outer products of the score vectors to calculate FIM
    FIM_y = Grad_y_vec*Grad_y_vec'/n;
    FIM_all(:, i) = FIM_y(:);

    % score this FIM
    sc = feval(score_fn, FIM_y);
    
    Sc(i) = sc;
    if sc_FIM_max < sc
        A_max = a;
        B_max = b;
        sc_FIM_max = sc;
    end

end
