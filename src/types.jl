
"""
simulate!(model::MultivariateModel)

Simulate a stochastic volatility model given m, P, and S. If P > 0, the mean vector is simulated
as a random walk while P = 0 implies a standard stochastic volatility model without time-varying mean(s).
"""

abstract type SSM end

mutable struct StateSpace <: SSM
    y::Matrix{<:Real} # T x J (dependent variables)
    x::Matrix{<:Real} # T x K (independent variables)

    m::Matrix{<:Real} # K x J (parameter means)
    P::Matrix{<:Real} # K x K (parameter covariance)
    S::Matrix{<:Real} # J x J ()

    β::Real # Discount factor for covariance matrix
    δ::Real # Discount factor for parameters
    ν::Real # Degrees of freedom of student-t distribution (calculated from β)
    n::Real # First shape parameter (input to)
    k::Real # Ensures that Σ is a random walk

    function StateSpace(y, x, m, P, S, β, δ)
        yrow, ycol = size(y)
        xrow, xcol = size(x)
        mrow, mcol = size(m)
        Prow, Pcol = size(P)
        Srow, Scol = size(S)

        !(yrow == xrow) && throw(DimensionMismatch("y is $(size(y)) and x is $(size(x))."))
        !(ycol == mcol == Srow == Scol) && throw(DimensionMismatch("y is $(size(y)), m is $(size(m)) and S is $(size(S))."))
        !(xcol == mrow == Prow == Pcol) && throw(DimensionMismatch("x is $(size(x)), m is $(size(m)) and P is $(size(P))."))

        all(LinearAlgebra.diag(P) .> 0) || throw(ArgumentError("All diagonal elements of P must be strictly positive."))
        all(LinearAlgebra.diag(S) .> 0) || throw(ArgumentError("All diagonal elements of S must be strictly positive."))

        (δ  > 0 && δ <= 1) || throw(ArgumentError("0 < δ ≤ 1 is not fulfilled."))
        (β > 2/3 && β < 1) || throw(ArgumentError("$(2//3) < β < 1 is not fulfilled."))
        ν = β/(1 - β)
        n = 1/(1 - β)
        k = (β - ycol*β + ycol)/(2β - ycol*β + ycol - 1)
        return new(y, x, m, P, S, β, δ, ν, n, k)
    end
end

# Constructor for state space model with default settings
function StateSpace(y::Matrix{<:Real}, x::Matrix{<:Real}, β::Real, δ::Real)
    J = size(y, 2)
    K = size(x, 2)
    StateSpace(y, x, zeros(K, J), Matrix(1000*LinearAlgebra.I, K, K), Matrix(LinearAlgebra.I, J, J), β, δ)
end

# Constructors for local level
function LocalLevel(y::Matrix{<:Real}, m::Matrix{<:Real}, P::Matrix{<:Real}, S::Matrix{<:Real}, β::Real, δ::Real)
    x = ones(size(y, 1), 1)
    StateSpace(y, x, m, P, S, β, δ)
end
function LocalLevel(y::Matrix{<:Real}, β::Real, δ::Real)
    J = size(y, 2)
    K = size(x, 2)
    x = ones(size(y, 1), 1)
    StateSpace(y, x, zeros(K, J), Matrix(1000*LinearAlgebra.I, K, K), Matrix(LinearAlgebra.I, J, J), β, δ)
end

function LocalLevel(y::Matrix{<:Real}, m::Matrix{<:Real}, P::Matrix{<:Real}, S::Matrix{<:Real}, β::Real, δ::Real)
    x = ones(size(y, 1), 1)
    StateSpace(y, x, m, P, S, β, δ)
end

# Constructors for local level trend
function LocalLevelTrend(y::Matrix{<:Real}, m::Matrix{<:Real}, P::Matrix{<:Real}, S::Matrix{<:Real}, β::Real, δ::Real)
    x = [ones(size(y, 1)) collect(1:size(y, 1))]
    StateSpace(y, x, m, P, S, β, δ)
end
function LocalLevelTrend(y::Matrix{<:Real}, β::Real, δ::Real)
    J = size(y, 2)
    K = size(x, 2)
    x = [ones(size(y, 1)) collect(1:size(y, 1))]
    StateSpace(y, x, zeros(K, J), Matrix(1000*LinearAlgebra.I, K, K), Matrix(LinearAlgebra.I, J, J), β, δ)
end