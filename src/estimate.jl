
"""
estimate(x; β::Real, δ::Real)

Estimate a stochastic volatility model with time-varying
mean and volatility. Volatility is modelled as a random walk.

The hyperparameters 2/3 < β < 1 and 0 < δ ≤ 1 are discount factors, controlling the shocks to the Σ and Ω respectively.
"""

function estimate(ssm::StateSpaceModel)
    y, F, G, β, k, m, P, S, β, δ, Δ, ν, n, k, T, p, d = ssm.y, ssm.F, ssm.G, ssm.β, ssm.k, ssm.m, ssm.P, ssm.S, ssm.β, ssm.δ, ssm.Δ, ssm.ν, ssm.n, ssm.k, ssm.T, ssm.p, ssm.d

    predicted = (μ = zeros(T, p), Σ = zeros(T, p, p), e = zeros(T, p), u = zeros(T, p), m = zeros(T, d, p), R = zeros(T, d, d), S = zeros(T, p, p))
    filtered  = (μ = zeros(T, p), Σ = zeros(T, p, p), e = zeros(T, p), u = zeros(T, p), m = zeros(T, d, p), P = zeros(T, d, d), S = zeros(T, p, p))

    for t = 1:T
        # Predict at time t-1
        R = get_R(P, G, Δ)
        Q = get_Q(F, R)
        μ = m' * G' * F
        e = y[t, :] - μ
        Σ = Q * (1 - β) / (3β*k - 2k) * S
        u = inv(cholesky(Σ).L) * e

        predicted.μ[t, :]    = μ
        predicted.Σ[t, :, :] = Σ
        predicted.e[t, :, :] = e
        predicted.u[t, :, :] = u
        predicted.m[t, :, :] = m
        predicted.R[t, :, :] = R
        predicted.S[t, :, :] = S

        # Update at time t
        K = get_K(F, R, Q)
        m = get_m(m, G, K, e)
        P = get_P(R, K, Q)
        S = get_S(S, k, e, Q)
        μ = m' * F
        e = y[t, :] - μ
        Σ = Q * (1 - β) / (2β - 1) * S
        u = inv(cholesky(Σ).L) * e

        filtered.μ[t, :]    = μ
        filtered.Σ[t, :, :] = Σ
        filtered.e[t, :, :] = e
        filtered.u[t, :, :] = u
        filtered.m[t, :, :] = m
        filtered.P[t, :, :] = P
        filtered.S[t, :, :] = S
    end
    return (predicted = predicted, filtered = filtered)
end