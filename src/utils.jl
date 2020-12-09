prior_covariance(S, P, Δ) = kron(S * (Δ * P * Δ))
posterior_covariance(S, P) = kron(S * P)