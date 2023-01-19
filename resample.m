function [res_ind] = resample(w, type, extra_params)

% [res_ind] = resample(w, type, rand_perm)

% This function performs resampling given the weights and the type of resampling. It is assumed that the indices (1:N) have weights w(1), w(2), ..., w(N) before resampling.
% There are three options for "type":
% 1. 'multinomial' performs conditional multinomial resampling
% 2. 'systematic' performs conditional systematic resampling
% 3. 'residual' performs conditional residual resampling

% Sinan Yildirim, 10.11.2014
% Last update: 10.11.2014, 18.34


% make w a normalised column vector:
w = w(:); w = w/sum(w);
N = length(w);

if strcmp(type, 'multinomial') == 1
    if nargin == 2
        M = N;  
    else
        M = extra_params.multinom_N;
    end
    res_ind = zeros(1, M);
    % generate ordered uniform random variables
    r_temp = rand(1, M);
    % u = exp(fliplr(cumsum((1./(M:-1:1)).*log(r_temp))));
    u = exp(cumsum((1./(M:-1:1)).*log(r_temp)));
    
    % cumulative distribution
    w_cum = cumsum(w);
    j = 1;
    for i = 1:M
        while w_cum(j) < u(M-i+1)
        	j = j + 1;
        end
        res_ind(i) = j;
    end
elseif strcmp(type, 'systematic') == 1
    res_ind = zeros(1, N);
    % Step 1: Draw uniform sample:
    u = rand;
    % Step 2: Perform systematic resampling using u
    w_cum = cumsum(w*N);
    j = 1;
    for i = 1:N
        while w_cum(j) < u
        	j = j + 1;
        end
        res_ind(i) = j;
        u = u + 1;
    end
elseif strcmp(type, 'residual') == 1
    res_ind = zeros(1, N);
    floor_w = floor(N*w);
    r = N*w - floor_w;
    ind_cum = 0;
    for i = 1:N
        res_ind(ind_cum+1:ind_cum + floor_w(i)) = i*ones(floor_w(i), 1);
        ind_cum = ind_cum + floor_w(i);
    end
    % Resample the rest of the indices from multinomial resampling
    % generate ordered uniform random variables
    R = N - ind_cum;
    extra_params.multinom_N = R;
    res_ind(ind_cum + 1:end) = resample(r, 'multinomial', extra_params);
end