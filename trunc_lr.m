function y = trunc_lr(x, a, b)

% y = trunc_lr(x, a, b)
%
% truncates x into [a, b]
% 
% Sinan Yildirim
% 19.01.2023

y = min(b, max(a, x));
