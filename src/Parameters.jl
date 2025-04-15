abstract type AbstractParameters end

mutable struct UnivariateParameters <: AbstractParameters
    m::Float64
    P::Float64
    S::Float64
    μ::Float64
    Σ::Float64
    const hp::Hyperparameters
    const p::Integer
    const k::Float64

    function UnivariateParameters(priors::UnivariatePriors, hp::Hyperparameters)
        p = priors.p
        k = get_k(hp, p)
        P = priors.P / hp.δ
        S = priors.S / k
        Σ = (P + 1) * hp.κ * S

        new(priors.m, P, S, deepcopy(priors.m), Σ, hp, p, k)
    end
end

mutable struct MultivariateParameters <: AbstractParameters
    m::Vector{Float64}
    P::Float64
    S::Matrix{Float64}
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    const hp::Hyperparameters
    const p::Integer
    const k::Float64

    function MultivariateParameters(priors::MultivariatePriors, hp::Hyperparameters)
        p = priors.p
        k = get_k(hp, p)
        P = priors.P / hp.δ
        S = priors.S / k
        Σ = (P + 1) * hp.κ * S

        new(priors.m, P, S, deepcopy(priors.m), Σ, hp, p, k)
    end
end