abstract type Parameters end

mutable struct Priors <: Parameters
    m::Vector{Float64}
    R::Float64
    S::Matrix{Float64}

    function Priors(m::AbstractVector,
                    R::Real,
                    S::AbstractMatrix)

        !isposdef(S) && throw(ArgumentError("S is not positive definite.")) 

        str = "m is a $(length(m))-dimensional vector, but S is a $(size(S, 1)) x $(size(S, 2)) matrix."
        !(length(m) == size(S, 1)) && throw(DimensionMismatch(str))
    
        any(R <= 0) && throw(ArgumentError("R must be strictly positive."))

        m = convert(Vector{Float64}, m)
        R = convert(Float64, R)
        S = convert(Matrix{Float64}, S)

        new(m, R, S)
    end
end

mutable struct Posteriors <: Parameters
    m::Vector{Float64}
    P::Float64
    S::Matrix{Float64}

    function Posteriors(m::AbstractVector,
                        P::Real,
                        S::AbstractMatrix)

        !isposdef(S) && throw(ArgumentError("S is not positive definite.")) 

        str = "m is a $(length(m))-dimensional vector, but S is a $(size(S, 1)) x $(size(S, 2)) matrix."
        !(length(m) == size(S, 1)) && throw(DimensionMismatch(str))

        any(P <= 0) && throw(ArgumentError("P must be strictly positive."))

        m = convert(Vector{Float64}, m)
        P = convert(Float64, P)
        S = convert(Matrix{Float64}, S)

        new(m, P, S)
    end
end

# Outer constructors
function Priors(p::Integer)
    @assert p > 0 "The input 'p' must be a strictly positive integer"

    m = zeros(p)
    P = 1000.0
    S = Diagonal(ones(p))

    return Priors(m, P, S)
end

get_p(param::Parameters) = size(param.S, 1)