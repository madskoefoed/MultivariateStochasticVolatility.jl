abstract type AbstractPriors end

struct UnivariatePriors <: AbstractPriors
    m::Float64
    P::Float64
    S::Float64
    p::Integer

    function UnivariatePriors(m::Real, P::Real, S::Real)
        any(P <= 0) && throw(ArgumentError("P must be strictly positive."))
        any(S <= 0) && throw(ArgumentError("S must be strictly positive."))

        m = convert(Float64, m)
        P = convert(Float64, P)
        S = convert(Float64, S)

        new(m, P, S, 1)
    end
end

struct MultivariatePriors <: AbstractPriors
    m::Vector{Float64}
    P::Float64
    S::Matrix{Float64}
    p::Integer

    function MultivariatePriors(m::AbstractVector, P::Real, S::AbstractMatrix)
        !isposdef(S) && throw(ArgumentError("S is not positive definite.")) 
        p = size(S, 1)

        str = "m is a $(length(m))-dimensional vector, but S is a $(size(S, 1)) x $(size(S, 2)) matrix."
        !(length(m) == size(S, 1)) && throw(DimensionMismatch(str))

        any(P <= 0) && throw(ArgumentError("P must be strictly positive."))

        m = convert(Vector{Float64}, m)
        P = convert(Float64, P)
        S = convert(Matrix{Float64}, S)

        new(m, P, S, p)
    end
end

##########################
### Outer constructors ###
##########################

Priors(m::Real, P::Real, S::Real)                     = UnivariatePriors(m, P, S)
Priors(m::AbstractVector, P::Real, S::AbstractMatrix) = MultivariatePriors(m, P, S)