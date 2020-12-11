
"""
estimate(x; β::Real, δ::Real)

Estimate a stochastic volatility model with time-varying
mean and volatility. Volatility is modelled as a random walk.

The hyperparameters 2/3 < β < 1 and 0 < δ ≤ 1 are discount factors, controlling the shocks to the Σ and Ω respectively.
"""

function estimate(ssm::StateSpace)

    y = ssm.y
    x = ssm.x

    β = ssm.β
    δ = ssm.δ
    ν = ssm.ν
    k = ssm.k
    n = ssm.n

    # Constants
    T, J = size(y)
    D    = size(x, 2)
    Δ = Matrix(LinearAlgebra.I, D, D)/sqrt(δ)

    m = zeros(T + 1, D, J)
    P = zeros(T + 1, D, D)
    S = zeros(T + 1, J, J)
    Φ = zeros(T + 1, D*J, D*J)

    μ = zeros(T, J)
    Σ = zeros(T, J, J)

    m[1, :, :] = ssm.m
    P[1, :, :] = ssm.P
    S[1, :, :] = ssm.S
    Φ[1, :, :] = posterior_covariance(ssm.S, ssm.P)

    for t = 1:T
        R = Δ * P[t, :, :] * Δ
        Q = x[t, :]' * R * x[t, :] + 1.0
        K = R * x[t, :] / Q

        μ[t, :]    = m[t, :, :]' * x[t, :]
        Σ[t, :, :] = Q * (1 - β) / (3β*k - 2k) * S[t, :, :]

        e = y[t, :] - μ[t, :]

        m[t + 1, :, :] = m[t, :, :] + K * e'
        P[t + 1, :, :] = R - K * K' * Q
        S[t + 1, :, :] = S[t, :, :] / k + e * e' / Q
    end
    return (μ = μ, Σ = Σ, m = m, P = P, S = S, Φ = Φ, k, Δ)
end