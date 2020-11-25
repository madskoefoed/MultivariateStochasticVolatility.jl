"""

"""

get_n(β) = 1/(1 - β)
get_k(β, p) = (β*(1 - p) + p)/(β*(2 - p) + p - 1)
get_df(β, n) = β*n
get_β(ν) = ν/(1 + ν) # Use degrees of freedom to get β

predictive_mean(m) = m
predictive_error(x, m) = x - m
predictive_covariance(Q, S, β, k) = (Q * (1 - β)) / (k * (3 * β - 2)) * S

update_m(m, K, e) = m + K * e
update_P(R, K, Q) = R - K^2 * Q
update_S(S, Q, e, k) = S / k + e^2 / Q
