
"""
estimate(x; β::Real, δ::Real)

Estimate a stochastic volatility model with time-varying
mean and volatility. Volatility is modelled as a random walk.

The hyperparameters 2/3 < β < 1 and 0 < δ ≤ 1 are discount factors, controlling the shocks to the Σ and Ω respectively.
"""

function estimate(TVVAR::UnivariateModel)

    x = TVVAR.x
    β = TVVAR.hyperparameters.β
    δ = TVVAR.hyperparameters.δ

    # Constants
    T = length(x)
    m = zeros(T + 1)
    P = zeros(T + 1)
    S = zeros(T + 1)

    m[1] = TVVAR.m
    P[1] = TVVAR.P
    S[1] = TVVAR.S

    μ = zeros(T)
    ϵ = zeros(T)
    Σ = zeros(T)

    for t = 1:T
        R, Q, K = update_state(P[t], δ)

        μ[t] = m[t]
        Σ[t] = Q*(1 - β)/(3 - 2/β)*S[t]
        ϵ[t] = x[t] - μ[t]

        m[t + 1] = m[t] + K*ϵ[t]
        P[t + 1] = R - K^2*Q
        S[t + 1] = S[t]*β + ϵ[t]^2/Q
    end
    return (μ = μ, Σ = Σ, m = m, P = P, S = S, Ω = S .* P)
end

function estimate(TVVAR::MultivariateModel)

    x = TVVAR.x
    β = TVVAR.hyperparameters.β
    δ = TVVAR.hyperparameters.δ
    ν = TVVAR.hyperparameters.ν

    # Constants
    T, p = size(x)
    k = (β - p*β + p)/(2β - p*β + p - 1)

    m = zeros(T + 1, p)
    P = zeros(T + 1)
    S = zeros(T + 1, p, p)

    m[1, :]    = TVVAR.m
    P[1]       = TVVAR.P
    S[1, :, :] = TVVAR.S

    μ = zeros(T, p)
    ϵ = zeros(T, p)
    Σ = zeros(T, p, p)

    for t = 1:T
        R, Q, K = update_state(P[t], δ)

        μ[t, :]    = m[t, :]
        Σ[t, :, :] = Q*(1 - β)/(3β*k - 2k)*S[t, :, :]
        ϵ[t, :]    = x[t, :] - μ[t, :]

        m[t + 1, :]    = m[t, :] + K*ϵ[t, :]
        P[t + 1]       = R - K^2*Q
        S[t + 1, :, :] = S[t, :, :]/k + ϵ[t, :]*ϵ[t, :]'/Q
    end
    return (μ = μ, Σ = Σ, m = m, P = P, S = S, Ω = S .* P)
end

T = 2000
x = randn(T) * 2;
h = Hyperparameters(0.98, 0.98)
m = UnivariateModel(x, 0, 1000, 1, h);
e = estimate(m);
plot!(sqrt.(e.Σ) * 1.96, color = :yellow)
plot!(-sqrt.(e.Σ) * 1.96, color = :yellow)
