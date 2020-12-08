
"""
simulate!(model::MultivariateModel)

Simulate a stochastic volatility model given m, P, and S. If P > 0, the mean vector is simulated
as a random walk while P = 0 implies a standard stochastic volatility model without time-varying mean(s).
"""

abstract type TVVAR end

mutable struct Hyperparameters
    β::Real
    δ::Real
    ν::Real
    function Hyperparameters(β, δ)
        (δ > 0 && δ <= 1) || throw(ArgumentError("0 < δ ≤ 1 is not fulfilled."))
        (β > 2/3 && β < 1) || throw(ArgumentError("$(2//3) < β < 1 is not fulfilled."))
        ν = β/(1 -β)
        return new(β, δ, ν)
    end
end

mutable struct UnivariateModel <: TVVAR
    x::AbstractVector{<:Real}
    m::Real
    P::Real
    S::Real
    hyperparameters::Hyperparameters
    function UnivariateModel(x, m, P, S, hyperparameters)
        P >= 0 || throw(ArgumentError("P must be non-negative."))
        S  > 0 || throw(ArgumentError("S must be strictly positive."))
        return new(x, m, P, S, hyperparameters)
    end
end

mutable struct MultivariateModel <: TVVAR
    x::AbstractMatrix{<:Real}
    m::AbstractVector{<:Real}
    P::Real
    S::AbstractMatrix{<:Real}
    hyperparameters::Hyperparameters
    function MultivariateModel(x, m, P, S, hyperparameters)
        !(length(m) == size(S, 1) == size(S, 2) == size(x, 2)) && throw(DimensionMismatch("x is $(size(x)), m is $(length(m)), and S is $(size(S))."))
        P >= 0 || throw(ArgumentError("P must be non-negative."))
        all(diag(S) .> 0) || throw(ArgumentError("The diagonal elements of S must be strictly positive."))
        return new(x, m, P, S, hyperparameters)
    end
end