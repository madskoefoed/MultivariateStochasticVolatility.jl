struct Priors
    m::Vector{Float64}
    P::Float64
    S::Matrix{Float64}

    function Priors(m::AbstractVector, P::Real, S::AbstractMatrix)
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

mutable struct Parameters
    m::Vector{Float64}
    P::Float64
    S::Matrix{Float64}
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    const hyper::Hyperparameters
    const p::Integer
    const k::Float64

    function Parameters(priors::Priors, hyper::Hyperparameters)
        p = size(priors.S, 1)
        k = get_k(hyper, p)

        P = priors.P / hyper.δ
        S = priors.S / k

        Σ = (P + 1) * hyper.κ * S

        new(priors.m, P, S, deepcopy(priors.m), Σ, hyper, p, k)
    end
end