
"""
estimate(x; β::Real, δ::Real)

Estimate a stochastic volatility model with time-varying
mean and volatility. Volatility is modelled as a random walk.

The hyperparameters 2/3 < β < 1 and 0 < δ ≤ 1 are discount factors, controlling the shocks to the Σ and Ω respectively.
"""

function estimate(ssm::StateSpace)

    y = ssm.y
    F = ssm.F
    G = ssm.G

    β = ssm.β
    #ν = ssm.ν
    k = ssm.k
    #n = ssm.n

    # Constants
    T, p = size(y)
    d    = size(F, 2)
    Δ = ones(d, d)/sqrt(ssm.δ)

    m = zeros(T + 1, d, p)
    P = zeros(T + 1, d, d)
    S = zeros(T + 1, p, p)
    Φ = zeros(T + 1, d*p, d*p)

    μ = zeros(T, p)
    e = zeros(T, p)
    u = zeros(T, p)
    Σ = zeros(T, p, p)

    m[1, :, :] = ssm.m
    P[1, :, :] = ssm.P
    S[1, :, :] = ssm.S
    Φ[1, :, :] = posterior_covariance(ssm.S, ssm.P)

    for t = 1:T
        R, Q, K = state_predict(F[t, :], G, P[t, :, :], Δ)

        μ[t, :]    = output_mean(F[t, :], G, m[t, :, :])
        Σ[t, :, :] = output_covariance(Q, S[t, :, :], β, k)

        μ[t, :], Σ[t, :, :] = output_predict(F[t, :], G, Q, m[t, :, :], S[t, :, :], β, k)

        e[t, :] = y[t, :] - μ[t, :]
        u[t, :] = standardized_error(y[t, :], μ[t, :], Σ[t, :, :])

        m[t + 1, :, :], P[t + 1, :, :], S[t + 1, :, :] = state_update(G, K, Q, R, m[t, :, :], S[t, :, :], e[t, :], k)

        Φ[t + 1, :, :] = posterior_covariance(S[t + 1, :, :], P[t + 1, :, :])

        #m[t + 1, :, :] = G * m[t, :, :] + K * e[t, :]'
        #P[t + 1, :, :] = R - K * K' * Q
        #S[t + 1, :, :] = S[t, :, :] / k + e[t, :] * e[t, :]' / Q
    end
    return (μ = μ, Σ = Σ, e = e, u = u, m = m, P = P, S = S, Φ = Φ, k, Δ)
end