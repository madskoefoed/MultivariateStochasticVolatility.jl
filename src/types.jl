
"""
    SVModel

Abstract type for a state space model with stochastic volatility.
"""

abstract type SVModel end

"""
    UnivariateModel

Definition of ``y, m, P, S`` for a univariate stochastic volatility (SV) model:

```math
\begin{gather*}
    \begin{aligned}
        y_{t} = \phi_{t} + \sigma_{t} \times \epsilon where \epsilon_{t} \tilde N(0, 1)
    \end{aligned}
\end{gather*}
```
"""

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

"""
    MultivariateModel

Definition of ``y, m, P, S`` for a multivariate stochastic volatility (SV) model:

```math
\begin{gather*}
    \begin{aligned}
        y_{t} = \phi_{t} + \Sigma_{t}^{frac{1}{2}} \times \epsilon_{t} where \epsilon_{t} \tilde N_{p}(0, I_{p})
    \end{aligned}
\end{gather*}
```
"""

mutable struct MultivariateModel{T<:AbstractFloat} <: SVModel
    y::AbstractMatrix{T}
    m::AbstractVector{T}
    P::T
    S::AbstractMatrix{T}
    function MultivariateModel(y::AbstractMatrix{T}, m::AbstractVector{T}, P::T, S::AbstractMatrix{T}) where {T<:AbstractFloat}
        !(size(y, 2) == length(m) == size(S, 1) == size(S, 2)) && throw(DimensionMismatch("y is $(size(y)), m is $(length(m)), and S is $(size(S))."))
        P > 0 || throw(ArgumentError("P must be strictly positive."))
        all(diag(S) .> 0) || throw(ArgumentError("The diagonal elements of S must be strictly positive."))
        return new{T}(y, m, P, S)
    end
end
