
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
    δ = ssm.δ
    ν = ssm.ν
    k = ssm.k
    n = ssm.n

    # Constants
    T, p = size(y)
    d    = size(F, 2)
    Δ = ones(d, d)/sqrt(δ)

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
        R = Δ * G * P[t, :, :] * G' * Δ
        Q = F[t, :]' * R * F[t, :] + 1.0
        K = R * F[t, :] / Q

        μ[t, :]    = m[t, :, :]' * G' * F[t, :]
        Σ[t, :, :] = Q * (1 - β) / (3β*k - 2k) * S[t, :, :]

        e[t, :] = y[t, :] - μ[t, :]
        u[t, :] = standardized_error(y[t, :], μ[t, :], Σ[t, :, :])

        m[t + 1, :, :] = G * m[t, :, :] + K * e[t, :]'
        P[t + 1, :, :] = R - K * K' * Q
        S[t + 1, :, :] = S[t, :, :] / k + e[t, :] * e[t, :]' / Q
    end
    return (μ = μ, Σ = Σ, e = e, u = u, m = m, P = P, S = S, Φ = Φ, k, Δ)
end