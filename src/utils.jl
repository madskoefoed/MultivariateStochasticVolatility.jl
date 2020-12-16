

prior_covariance(S, P, Δ) = LinearAlgebra.kron(S, (Δ * P * Δ))
posterior_covariance(S, P) = LinearAlgebra.kron(S, P)

function standardized_error(y, μ, Σ)
    T, p = size(y)
    e = y - μ
    for t in 1:T
        e[t, :] = inv(cholesky(Σ))*e[t, :]
    end
    return e
end