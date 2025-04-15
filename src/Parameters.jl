mutable struct Parameters
    m::Vector{Float64}
    P::Float64
    S::Matrix{Float64}
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    const hp::Hyperparameters
    const p::Integer
    const k::Float64

    function Parameters(priors::Priors, hp::Hyperparameters)
        p = size(priors.S, 1)
        k = get_k(hp, p)

        P = priors.P / hp.δ
        S = priors.S / k

        Σ = (P + 1) * hp.κ * S

        new(priors.m, P, S, deepcopy(priors.m), Σ, hp, p, k)
    end
end