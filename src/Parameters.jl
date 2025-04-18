abstract type AbstractParameters end

mutable struct MeanParameters <: AbstractParameters
    m::Vector{Float64}
    P::Float64
    S::Matrix{Float64}
    const hyper::Hyperparameters
    const p::Integer
    const k::Float64

    function MeanParameters(m::AbstractVector,
                            P::Real,
                            S::AbstractMatrix,
                            hyper::Hyperparameters)

        isposdef(S) || throw(ArgumentError("S is not positive definite."))

        str = "m is a $(length(m))-dimensional vector, but S is a $(size(S, 1)) x $(size(S, 2)) matrix."
        !(length(m) == size(S, 1)) && throw(DimensionMismatch(str))

        any(P <= 0) && throw(ArgumentError("P must be strictly positive."))

        m = convert(Vector{Float64}, m)
        P = convert(Float64, P)
        S = convert(Matrix{Float64}, S)

        p = size(S, 1)
        k = get_k(hyper, p)

        new(m, P, S, hyper, p, k)
    end
end