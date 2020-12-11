
"""
simulate!(model::MultivariateModel)

Simulate a stochastic volatility model given m, P, and S. If P > 0, the mean vector is simulated
as a random walk while P = 0 implies a standard stochastic volatility model without time-varying mean(s).
"""

abstract type SSM end

mutable struct Hyperparameters
    β::Real
    δ::Real
    ν::Real
    n::Real
    function Hyperparameters(β, δ)
        (δ  > 0 && δ <= 1) || throw(ArgumentError("0 < δ ≤ 1 is not fulfilled."))
        (β > 2/3 && β < 1) || throw(ArgumentError("$(2//3) < β < 1 is not fulfilled."))
        ν = β/(1 - β)
        n = 1/(1 - β)
        return new(β, δ, ν, n)
    end
end

# Constructors for Hyperparameters
Hyperparameters() = Hyperparameters(0.999, 0.999)

mutable struct Priors
    m::Matrix{<:Real} # K x J
    P::Matrix{<:Real} # K x K
    S::Matrix{<:Real} # J x J
    function Priors(m, P, S)
        mrow, mcol = size(m)
        Prow, Pcol = size(P)
        Srow, Scol = size(S)

        !(size(m, 2) == size(S, 1) == size(S, 2)) && throw(DimensionMismatch("m is $(size(m)) and S is $(size(S))."))
        !(size(m, 1) == size(P, 1) == size(P, 2)) && throw(DimensionMismatch("m is $(size(m)) and P is $(size(P))."))
        return new(m, P, S)
    end
end

# Constructors for Priors
#Priors(m::Real, P::Real, S::Real) = Priors(num2mat(m), num2mat(P), num2mat(S))
#Priors(m::Real, P::Matrix{<:Real}, S::Matrix{<:Real}) = Priors(num2mat(m), P, S)
#Priors(m::Real, P::Matrix{<:Real}, S::Matrix{<:Real}) = Priors(num2mat(m), P, S)
#Priors(m::Real, P::Matrix{<:Real}, S::Matrix{<:Real}) = Priors(num2mat(m), P, S)

#Priors(m::Vector{<:Real}, P::Matrix{<:Real}, S::Matrix{<:Real}) = Priors(vec2mat(m), P, S)
#Priors(m::Matrix{<:Real}, P::Vector{<:Real}, S::Matrix{<:Real}) = Priors(m, vec2mat(P), S)
#Priors(m::Matrix{<:Real}, P::Matrix{<:Real}, S::Vector{<:Real}) = Priors(m, P, repvec2mateat(S))

mutable struct StateSpace <: SSM
    y::Matrix{<:Real} # T x J
    x::Matrix{<:Real} # T x K
    priors::Priors
    hyperparameters::Hyperparameters
    function StateSpace(y, x, priors, hyperparameters)
        yrow, ycol = size(y)
        xrow, xcol = size(x)
        mrow, mcol = size(priors.m)
        Prow, Pcol = size(priors.P)
        Srow, Scol = size(priors.S)

        !(yrow == xrow) && throw(DimensionMismatch("y is $(size(y)) and x is $(size(x))."))
        !(ycol == mcol == Srow == Scol) && throw(DimensionMismatch("y is $(size(y)), m is $(size(m)) and S is $(size(S))."))
        !(xcol == mrow == Prow == Pcol) && throw(DimensionMismatch("x is $(size(x)), m is $(size(m)) and P is $(size(P))."))
        return new(y, x, priors, hyperparameters)
    end
end

LocalLevel(y::Vector{<:Real}, priors::Priors, hyperparameters::Hyperparameters) = StateSpace(vec2mat(y), ones(size(y, 1), 1), priors, hyperparameters)
LocalLevel(y::Matrix{<:Real}, priors::Priors, hyperparameters::Hyperparameters) = StateSpace(y, ones(size(y, 1), 1), priors, hyperparameters)

LocalLevelTrend(y::Vector{<:Real}, priors::Priors, hyperparameters::Hyperparameters) = StateSpace(vec2mat(y), [ones(size(y, 1)) collect(1:size(y, 1))], priors, hyperparameters)
LocalLevelTrend(y::Matrix{<:Real}, priors::Priors, hyperparameters::Hyperparameters) = StateSpace(y, [ones(size(y, 1)) collect(1:size(y, 1))], priors, hyperparameters)