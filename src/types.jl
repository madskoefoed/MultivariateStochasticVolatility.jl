
#@doc raw"""
#    StochasticVolatilityVModel
#Definition of the system matrices ``y, m, P, S`` for a stochastic volatility (SV) model.
#2/3 < β < 1 and 0 < δ ≤ 1 are discount factors, controlling the shocks to the XXX and XXX respectively. Setting δ = 1 results
# in a SV model without time-varying parameters.
#"""

abstract type SVModel end

mutable struct UnivariateModel{T<:AbstractFloat} <: SVModel
    y::AbstractVector{T}
    m::T
    P::T
    S::T
    function UnivariateModel(y::AbstractVector{T}, m::T, P::T, S::T) where {T<:AbstractFloat}
        P > 0 || throw(ArgumentError("P must be strictly positive."))
        S > 0 || throw(ArgumentError("S must be strictly positive."))
        return new{T}(y, m, P, S)
    end
end

mutable struct MultivariateModel{T<:AbstractFloat} <: SVModel
    y::AbstractMatrix{T}
    m::AbstractVector{T}
    P::T
    S::AbstractMatrix{T}
    function MultivariateModel(y::AbstractMatrix{T}, m::AbstractMatrix{T}, P::T, S::AbstractMatrix{T}) where {T<:AbstractFloat}
        !(size(y, 2) == length(m) == size(S, 1) == size(S, 2)) && throw(DimensionMismatch("y is $(size(y)), m is $(length(m)), and S is $(size(S))."))
        P > 0 || throw(ArgumentError("P must be strictly positive."))
        all(diag(S) .> 0) || throw(ArgumentError("The diagonal elements of S must be strictly positive."))
        return new{T}(y, m, P, S)
    end
end
