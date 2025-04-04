abstract type AbstractParameters end
abstract type AbstractStochVolParameters <: AbstractParameters end

struct StochvolParameters <: AbstractStochVolParameters
    m::Float64
    P::Float64
    S::Float64

    function StochvolParameters(m::Real,
                                P::Real,
                                S::Real)
        
        !isposdef(S) && throw(ArgumentError("S is not positive definite.")) 

        str = "m is a $(length(m))-dimensional vector, but S is a $(size(S, 1)) x $(size(S, 2)) matrix."
        !(length(m) == size(S, 1)) && throw(DimensionMismatch(str))
    
        P <= 0 && throw(ArgumentError("P must be strictly positive."))
        S <= 0 && throw(ArgumentError("S must be strictly positive."))

        m = convert(Float64, m)
        P = convert(Float64, P)
        S = convert(Float64, S)

        new(m, P, S)
    end
end

struct MvStochvolParameters <: AbstractStochVolParameters
    m::Vector{Float64}
    P::Float64
    S::Matrix{Float64}
    #μ::Vector{Float64}
    #Σ::Matrix{Float64}
    #hyper::Hyperparameters
    #p::Integer
    #k::Float64

    function MvStochvolParameters(m::AbstractVector,
                                  P::Real,
                                  S::AbstractMatrix)
        
        !isposdef(S) && throw(ArgumentError("S is not positive definite.")) 

        str = "m is a $(length(m))-dimensional vector, but S is a $(size(S, 1)) x $(size(S, 2)) matrix."
        !(length(m) == size(S, 1)) && throw(DimensionMismatch(str))
    
        any(P <= 0) && throw(ArgumentError("P must be strictly positive."))

        m = convert(Vector{Float64}, m)
        P = convert(Float64, P)
        S = convert(Matrix{Float64}, S)

        #p = size(S, 1)
        #k = get_k(hyper, p)

        #P = P / hyper.δ
        #S = S / k

        #Σ = (P + 1) * (1 - hyper.β) / (3*hyper.β - 2) * S

        new(m, P, S)
    end
end

### Outer constructors ###
StochvolParameters() = StochvolParameters(0.0, 1000.0, 1.0)

function MvStochvolParameters(p::Integer)
    @assert p > 0 "Input 'p' must be strictly positive."

    m = zeros(p)
    P = 1000.0
    S = Matrix(Diagonal(1000 * ones(p)))

    return MvStochvolParameters(m, P, S)
end

### Functions ###
get_p(param::AbstractParameters) = size(param.S, 1)