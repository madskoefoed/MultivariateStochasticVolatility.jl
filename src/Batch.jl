mutable struct MvStochVolBatch
    y::Matrix{<:Real}
    error::Matrix{Float64}
    scaled::Matrix{Float64}
    m::Matrix{Float64}
    P::Vector{Float64}
    S::Array{Float64, 3}
    μ::Matrix{Float64}
    Σ::Array{Float64, 3}
    nobs::Integer
    loglik::Float64
    const hyperparameters::Hyperparameters
    const p::Integer
    const k::AbstractFloat

    function MvStochVolBatch(m::AbstractVector,
                             P::Real,
                             S::AbstractMatrix,
                             h::Hyperparameters,
                             y::Matrix{<:Real})

    !isposdef(S) && throw(ArgumentError("S is not positive definite.")) 

    str = "m is a $(length(m))-dimensional vector, but S is a $(size(S, 1)) x $(size(S, 2)) matrix."
    !(length(m) == size(S, 1)) && throw(DimensionMismatch(str))

    any(P <= 0) && throw(ArgumentError("P must be strictly positive."))

    T, p = size(y)
    m = reshape(m, (T, p))
    P = convert(Float64, P)
    S = convert(Matrix{Float64}, S)

    μ = get_mean(m)
    Σ = get_covariance(P, S, h.κ)

    y = Vector{Float64}(undef, p)#fill(missing, p)
    e = Vector{Float64}(undef, p)#fill(missing, p)
    z = Vector{Float64}(undef, p)#fill(missing, p)
    k = get_k(h, p)

    nobs = 0
    ll   = 0.0

    new(y, e, z, m, P, S, μ, Σ, nobs, ll, h, p, k)
    end
end