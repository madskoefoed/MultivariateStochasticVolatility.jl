
"""
simulate!(model::MultivariateModel)

Simulate a stochastic volatility model given m, P, and S. If P > 0, the mean vector is simulated
as a random walk while P = 0 implies a standard stochastic volatility model without time-varying mean(s).

y' = F' * Θ + e'            t = Z * α + e
Θ = G * Θ + walk            α = T * α + R * η
"""

abstract type SSM end

mutable struct StateSpace <: SSM
    y::Matrix{<:Real} # T x p (response matrix)
    F::Matrix{<:Real} # T x d (design matrix)
    G::Matrix{<:Real} # d x d (evolution matrix)

    m::Matrix{<:Real} # d x p (parameter means)
    P::Matrix{<:Real} # d x d (parameter covariance)
    S::Matrix{<:Real} # p x p ()

    β::Real # Discount factor for covariance matrix
    δ::Real # Discount factor for parameters
    ν::Real # Degrees of freedom of student-t distribution (calculated from β)
    n::Real # First shape parameter (input to)
    k::Real # Ensures that Σ is a random walk

    function StateSpace(y, F, G, m, P, S, β, δ)
        yrow, ycol = size(y)
        Frow, Fcol = size(F)
        Grow, Gcol = size(G)
        mrow, mcol = size(m)
        Prow, Pcol = size(P)
        Srow, Scol = size(S)

        !(yrow == Frow) && throw(DimensionMismatch("y has $(size(y, 1)) rows and F has $(size(F, 1)) rows."))
        !(ycol == mcol == Srow == Scol) && throw(DimensionMismatch("y has $(size(y, 2)) columns, m has $(size(m)) columns, and S is a symmetric $(size(S)) matrix."))
        !(mrow == Prow == Pcol == Grow == Gcol) && throw(DimensionMismatch("F has $(size(F, 2)) columns, m has $(size(m, 1)) columns and P is a symmetric $(size(P)) matrix."))

        all(LinearAlgebra.diag(P) .> 0) || throw(ArgumentError("All diagonal elements of P must be strictly positive."))
        all(LinearAlgebra.diag(S) .> 0) || throw(ArgumentError("All diagonal elements of S must be strictly positive."))

        (δ  > 0 && δ <= 1) || throw(ArgumentError("0 < δ ≤ 1 is not fulfilled."))
        (β > 2/3 && β < 1) || throw(ArgumentError("$(2//3) < β < 1 is not fulfilled."))
        ν = β/(1 - β)
        n = 1/(1 - β)
        k = (β - ycol*β + ycol)/(2β - ycol*β + ycol - 1)
        return new(y, F, G, m, P, S, β, δ, ν, n, k)
    end
end

# Constructors for local level
function LocalLevel(y::Matrix{<:Real}, m::Matrix{<:Real}, P::Matrix{<:Real}, S::Matrix{<:Real}, β::Real, δ::Real)
    T, p = size(y)
    F = ones(T, 1)
    G = ones(1, 1)
    return StateSpace(y, F, G, m, P, S, β, δ)
end
function LocalLevel(y::Matrix{<:Real}, β::Real, δ::Real)
    T, p = size(y)
    F = ones(T, 1)
    G = ones(1, 1)
    m = zeros(1, p)
    P = ones(1, 1)*1000
    S = Matrix(LinearAlgebra.I, p, p)
    return StateSpace(y, F, G, m, P, S, β, δ)
end

function LocalLevel(y::Matrix{<:Real}, m::Matrix{<:Real}, P::Matrix{<:Real}, S::Matrix{<:Real}, β::Real, δ::Real)
    F = ones(size(y, 1), 1)
    G = ones(1, 1)
    return StateSpace(y, F, G, m, P, S, β, δ)
end

# Constructors for local level trend
function LocalLevelTrend(y::Matrix{<:Real}, m::Matrix{<:Real}, P::Matrix{<:Real}, S::Matrix{<:Real}, β::Real, δ::Real)
    F = [ones(size(y, 1)) collect(1:size(y, 1))]
    G = [1.0 1.0; 0.0 1.0]
    return StateSpace(y, F, G, m, P, S, β, δ)
end
function LocalLevelTrend(y::Matrix{<:Real}, β::Real, δ::Real)
    T, p = size(y)
    d = size(x, 2)
    F = [ones(T) collect(1:T)]
    G = [1.0 1.0; 0.0 1.0]
    m = zeros(2, p)
    P = Matrix(1000*LinearAlgebra.I, 2, 2)
    S = Matrix(LinearAlgebra.I, p, p)
    return StateSpace(y, F, G, m, P, S, β, δ)
end
#y' = F' * Θ + e'            t = Z * α + e
#Θ = G * Θ + walk            α = T * α + R * η