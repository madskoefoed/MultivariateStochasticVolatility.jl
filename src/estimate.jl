
"""
    estimate(model::UnivariateModel; β::Real, δ::Real)

Estimate a univariate stochastic volatility model with time-varying
mean and volatility.

The hyperparameters 2/3 < β < 1 and 0 < δ ≤ 1 are discount factors, controlling the shocks to the Σ and Ω respectively.
"""

function estimate(model::UnivariateModel; β::Real, δ::Real)

    y = model.y

    # Constants
    T = size(y, 1)
    n = get_n(β)
    k = get_k(β, 1)
    
    m = zeros(T + 1) ; m[1] = model.m
    P = zeros(T + 1) ; P[1] = model.P
    S = zeros(T + 1) ; S[1] = model.S

    μ = zeros(T)
    ϵ = zeros(T)
    Σ = zeros(T)

    for t = 1:T
        R = P[t] / δ
        Q = R + 1.0
        K = R / Q
        
        μ[t] = predictive_mean(m[t])
        Σ[t] = predictive_covariance(Q, S[t], β, k)
        ϵ[t] = predictive_error(y[t], μ[t])

        m[t + 1] = update_m(m[t], K, ϵ[t])
        P[t + 1] = update_P(R, K, Q)
        S[t + 1] = update_S(S[t], Q, ϵ[t], k)
    end
    return (μ = μ, Σ = Σ, ϵ = ϵ, ν = get_df(β, n), m = m, P = P, S = S)
end

"""
estimate(model::MultivariateModel; β::Real, δ::Real)

Estimate a multivariate stochastic volatility model with time-varying
mean and volatility.

The hyperparameters 2/3 < β < 1 and 0 < δ ≤ 1 are discount factors, controlling the shocks to the Σ and Ω respectively.
"""

function estimate(model::MultivariateModel; β::Real, δ::Real)

    y = model.y

    # Constants
    T, p = size(y)
    n = get_n(β)
    k = get_k(β, p)
    
    m = zeros(T + 1, p)    ; m[1, :]    = model.m
    P = zeros(T + 1)       ; P[1]       = model.P
    S = zeros(T + 1, p, p) ; S[1, :, :] = model.S

    μ = zeros(T, p)
    ϵ = zeros(T, p)
    Σ = zeros(T, p, p)

    for t = 1:T
        R = P[t] / δ
        Q = R + 1.0
        K = R / Q

        μ[t, :]    = predictive_mean(m[t, :])
        Σ[t, :, :] = predictive_covariance.(Q, S[t, :, :], β, k)
        ϵ[t, :]    = predictive_error.(y[t, :], μ[t, :])

        m[t + 1, :]    = update_m.(m[t, :], K, ϵ[t, :])
        P[t + 1]       = update_P(R, K, Q)
        S[t + 1, :, :] = update_S.(S[t, :, :], Q, ϵ[t, :], k)
    end
    return (μ = μ, Σ = Σ, ϵ = ϵ, ν = get_df(β, n), m = m, P = P, S = S)
end