
"""
estimate(x; β::Real, δ::Real)

Estimate a multivariate stochastic volatility model with time-varying
mean and volatility.

The hyperparameters 2/3 < β < 1 and 0 < δ ≤ 1 are discount factors, controlling the shocks to the Σ and Ω respectively.
"""

function estimate(x::AbstractMatrix{W}, S::AbstractMatrix{W}; ν::W) where {W<:Real}

    ν > 2 || throw(ArgumentError("ν must be > 2."))

    T, p = size(x)
    β = ν/(1 + ν)
    k = (β - p*β + p)/(2β - p*β + p - 1)
    
    Σ = zeros(T, p, p)
    s = copy(S)
    S = zeros(T + 1, p, p) ; S[1, :, :] = s
    for t = 1:T
        Σ[t, :, :] = (1 - β)/(3β*k - 2k) * S[t, :, :]
        S[t + 1, :, :] = S[t, :, :]/k + x[t, :]*x[t, :]'
    end
    if p == 1
        Σ = Σ[:, 1, 1]
        S = S[:, 1, 1]
    end
    return (Σ = Σ, ν = ν, S = S)
end
estimate(x::AbstractVector{W}, S::W; ν::W) where {W<:Real} = estimate(repeat(x, 1, 1), Matrix([S]'), ν = ν)
estimate(x::AbstractMatrix{W}, S::AbstractVector{W}; ν::W) where {W<:Real} = estimate(x, Diagonal(S), ν = ν)

function estimate(x::AbstractMatrix{W}, m::AbstractVector{W}, P::W, S::AbstractMatrix{W}; ν::W, δ::W) where {W<:Real}

    # Checks
    !(length(m) == size(S, 1) == size(S, 2)) && throw(DimensionMismatch("x is $(size(x)), m is $(length(m)), and S is $(size(S))."))
    ν > 2 || throw(ArgumentError("ν must be > 2."))
    P > 0 || throw(ArgumentError("P must be strictly positive."))
    all(diag(S) .> 0) || throw(ArgumentError("The diagonal elements of S must be strictly positive."))
    #!issymmetric(S) || throw(ArgumentError("S must be symmetric."))

    # Constants
    T, p = size(x)
    β = ν/(1 + ν)
    k = (β - p*β + p)/(2β - p*β + p - 1)

    s = copy(S)
    m = repeat(m', T + 1, 1)
    P = fill(P, T + 1)
    S = zeros(T + 1, p, p) ; S[1, :, :] = s

    μ = zeros(T, p)
    ϵ = zeros(T, p)
    Σ = zeros(T, p, p)

    for t = 1:T
        R = P[t]/δ
        Q = R + 1.0
        K = R/Q

        μ[t, :]    = m[t, :]
        Σ[t, :, :] = Q*(1 - β)/(3β*k - 2k)*S[t, :, :]
        ϵ[t, :]    = x[t, :] - μ[t, :]

        m[t + 1, :]    = m[t, :] + K*ϵ[t, :]
        P[t + 1]       = R - K^2*Q
        S[t + 1, :, :] = S[t, :, :]/k + ϵ[t, :]*ϵ[t, :]'/Q
    end
    if p == 1
        Σ = Σ[:, 1, 1]
        S = S[:, 1, 1]
    end
    return (μ = μ, Σ = Σ, ν = ν, m = m, P = P, S = S)
end
estimate(x::AbstractVector{W}, m::W, P::W, S::W; ν::W, δ::W) where {W<:Real} = estimate(repeat(x, 1, 1), [m], P, Matrix([S]'), ν = ν, δ = δ)
estimate(x::AbstractMatrix{W}, m::AbstractVector{W}, P::W, S::AbstractVector{W}; ν::W, δ::W) where {W<:Real} = estimate(x, m, P, Diagonal(S) ν = ν, δ = δ)

T = 200
x = randn(T)*3;
plot(x, color = :green)
m = estimate(x, 100.0; ν = 10.);
plot!(sqrt.(m.Σ) * 1.96, color = :red)
plot!(-sqrt.(m.Σ) * 1.96, color = :red)

M = estimate(x, 1.0, 1.0, 100.0; ν = 10., δ = 0.98);
plot!(sqrt.(M.Σ) * 1.96, color = :yellow)
plot!(-sqrt.(M.Σ) * 1.96, color = :yellow)

for i = 5:15
    T = 20_000
    x = randn(T)*10;
    x = rand(TDist(10.0), T)*10
    m = estimate(x, 1000.0; ν = convert(Float64, i));
    l = quantile(TDist(i), 0.05)
    u = quantile(TDist(i), 0.95)
    inside = l .< x ./ sqrt.(m.Σ) .< u
    inside = mean(inside[10_000:end])
    println("For a degrees of freedom = $i, we have coverage of $inside")
end