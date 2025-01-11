mutable struct MvStochVol
    m::Vector{Float64}
    P::Float64
    S::Matrix{Float64}
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    y::Vector{<:Real}
    error::Vector{Float64}
    scaled::Vector{Float64}
    loglik::Float64
    obs::Integer
    const hyperparameters::Hyperparameters
    const p::Integer
    const k::Float64

    function MvStochVol(m::AbstractVector,
                        P::Real,
                        S::AbstractMatrix,
                        hyper::Hyperparameters)
        
        !isposdef(S) && throw(ArgumentError("S is not positive definite.")) 

        str = "m is a $(length(m))-dimensional vector, but S is a $(size(S, 1)) x $(size(S, 2)) matrix."
        !(length(m) == size(S, 1)) && throw(DimensionMismatch(str))
    
        any(P <= 0) && throw(ArgumentError("P must be strictly positive."))

        m = convert(Vector{Float64}, m)
        P = convert(Float64, P)
        S = convert(Matrix{Float64}, S)

        p = size(S, 1)
        k = get_k(hyper, p)

        # Predict at time t=1 based on priors: m, P, S
        P = P / hyper.δ
        S = S/k

        μ = prior_mean(m)
        Σ = prior_covariance(P, S, hyper)
        l = 0.0
        o = 0

        # Unknown at time t=0
        y = e = z = zeros(p)

        new(m, P, S, μ, Σ, y, e, z, l, o, hyper, p, k)
    end
end

prior_mean(m::Vector{Float64}) = m
prior_covariance(P::Float64, S::Matrix{Float64}, hyper::Hyperparameters) = (P + 1) * (1 - hyper.β) / (3*hyper.β - 2) * S