mutable struct Parameters
    m::Vector{<:AbstractFloat}
    P::AbstractFloat
    S::AbstractMatrix

    function Parameters(m::Vector{<:Real},
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
function Parameters(p::Integer)
    @assert p > 0 "The input 'p' must be a strictly positive integer"

    m = zeros(p)
    P = 1000.0
    S = Diagonal(ones(p))

    return Parameters(m, P, S)
end