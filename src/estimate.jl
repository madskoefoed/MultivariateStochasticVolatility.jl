
"""
estimate(x; β::Real, δ::Real)

Estimate a multivariate stochastic volatility model with time-varying
mean and volatility.

The hyperparameters 2/3 < β < 1 and 0 < δ ≤ 1 are discount factors, controlling the shocks to the Σ and Ω respectively.
"""

function estimate(x::AbstractMatrix{<:Real}, m::AbstractVector{<:Real}, P::W, S::AbstractMatrix{<:Real}; ν::Real, δ::Real)

    # Checks
    !(length(m) == size(S, 1) == size(S, 2)) && throw(DimensionMismatch("x is $(size(x)), m is $(length(m)), and S is $(size(S))."))
    ν > 2 || throw(ArgumentError("ν must be > 2."))
    P >= 0 || throw(ArgumentError("P must be non-negative."))
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
    return (μ = μ, Σ = Σ, ν = ν, m = m, P = P, S = S)
end
estimate(x::AbstractVector{<:Real}, m::Real, P::Real, S::Real; ν::Real, δ::Real) = estimate(repeat(x, 1, 1), [m], P, Matrix([S]'), ν = ν, δ = δ)
estimate(x::AbstractMatrix{<:Real}, m::AbstractVector{<:Real}, P::Real, S::AbstractVector{<:Real}; ν::Real, δ::Real) = estimate(x, m, P, Diagonal(S), ν = ν, δ = δ)
estimate(x::AbstractMatrix{<:Real}, S::AbstractMatrix{<:Real}; ν::Real, δ::Real) = estimate(x, zeros(size(x, 2)), 0.0, S; ν = ν, δ = δ)
estimate(x::AbstractMatrix{<:Real}, P::Real, S::AbstractMatrix{<:Real}; ν::Real, δ::Real) = estimate(x, zeros(size(x, 2)), P, S; ν = ν, δ = δ)
estimate(x::AbstractMatrix{<:Real}, m::AbstractVector{<:Real}, S::AbstractMatrix{<:Real}; ν::Real, δ::Real) = estimate(x, m, 0.0, S; ν = ν, δ = δ)

T = 200
x = randn(T) * 2;
M = estimate(x, 0.0, 0.00001, 100.0; ν = 10., δ = 0.98);
plot!(sqrt.(M.Σ) * 1.96, color = :yellow)
plot!(-sqrt.(M.Σ) * 1.96, color = :yellow)
