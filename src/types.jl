
"""
simulate!(model::MultivariateModel)

Simulate a stochastic volatility model given m, P, and S. If P > 0, the mean vector is simulated
as a random walk while P = 0 implies a standard stochastic volatility model without time-varying mean(s).

y' = F' * Θ + e'            yt   = Zt * αt + et          et ~ N(0, Ht)
Θ = G * Θ + walk            αt+1 = Tt * αt + Rt * ηt     ηt ~ N(0, Qt)
"""

abstract type SSM end

mutable struct StateSpaceModel <: SSM
    y::Matrix{<:Real} # T x p (response matrix)
    F::Vector{<:Real} # d x 1 (design vector)
    G::Matrix{<:Real} # d x d (evolution matrix)

    m::Matrix{<:Real} # d x p (parameter means)
    P::Matrix{<:Real} # d x d (parameter covariance)
    S::Matrix{<:Real} # p x p ()

    β::Real # Discount factor for covariance matrix
    δ::Real # Discount factor for parameters
    Δ::Matrix{<:Real}
    ν::Real # Degrees of freedom of student-t distribution (calculated from β)
    n::Real # First shape parameter (input to)
    k::Real # Ensures that Σ is a random walk
    T::Integer
    p::Integer
    d::Integer

    function StateSpaceModel(y, F, G, m, P, S, β, δ)
        T = size(y, 1)
        p = size(y, 2)
        d = size(m, 1)

        # Check dimensions of the output equation
        !(p == size(m, 2) == size(S, 1) == size(S, 2)) && throw(DimensionMismatch("y has $p columns, m has $(size(m, 2)) columns, and S is a symmetric $(size(S)) matrix."))
        
        # Check dimensions of the state equation
        !(d == length(F) == size(P, 1) == size(P, 2) == size(G, 1) == size(G, 2)) && throw(DimensionMismatch("F has length $(length(F)), m has $(size(m, 1)) rows and P is a symmetric $(size(P)) matrix."))

        all(diag(P) .> 0) || throw(ArgumentError("All diagonal elements of P must be strictly positive."))
        all(diag(S) .> 0) || throw(ArgumentError("All diagonal elements of S must be strictly positive."))

        (δ  > 0 && δ <= 1) || throw(ArgumentError("0 < δ ≤ 1 is not fulfilled."))
        (β > 2/3 && β < 1) || throw(ArgumentError("$(2//3) < β < 1 is not fulfilled."))
        n = 1/(1 - β)
        ν = β*n
        k = (β - p*β + p)/(2β - p*β + p - 1)
        Δ = diagm(fill(1/δ, d))
        return new(y, F, G, m, P, S, β, δ, Δ, ν, n, k, T, p, d)
    end
end

# Constructors for local level
function LocalLevel(y::Matrix{<:Real}, m::Matrix{<:Real}, P::Matrix{<:Real}, S::Matrix{<:Real}, β::Real, δ::Real)
    F = [1]#ones(1, 1)
    G = ones(1, 1)
    return StateSpaceModel(y, F, G, m, P, S, β, δ)
end

# Constructors for local level trend
function LocalLevelTrend(y::Matrix{<:Real}, m::Matrix{<:Real}, P::Matrix{<:Real}, S::Matrix{<:Real}, β::Real, δ::Real)
    F = [1, 0]#[ones(1, 1); zeros(1, 1)]
    G = [1 1; 0 1]
    return StateSpaceModel(y, F, G, m, P, S, β, δ)
end

# Constructors for local level cycle
function LocalLevelCycle(y::Matrix{<:Real}, m::Matrix{<:Real}, P::Matrix{<:Real}, S::Matrix{<:Real}, β::Real, δ::Real)
    F = [1, 1, 0]#[ones(2, 1); zeros(1, 1)]
    G = [1  0 0; 0 0 0; 0 0 0]
    return StateSpaceModel(y, F, G, m, P, S, β, δ)
end