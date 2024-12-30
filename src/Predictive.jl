mutable struct PriorPredictive
    μ::Vector{<:AbstractFloat}
    Σ::AbstractMatrix

    function PriorPredictive(μ::Vector{<:AbstractFloat},
                        Σ::AbstractMatrix)

        str = "μ is a $(length(μ))-dimensional vector, but Σ is a $(size(Σ, 1)) x $(size(Σ, 2)) matrix."
        !(length(μ) == size(Σ, 1) == size(Σ, 2)) && throw(DimensionMismatch(str))
    
        new(μ, Σ)
    end
end

function PriorPredictive(priors::Priors, h::Hyperparameters)
    μ = get_mean(priors)
    Σ = get_covariance(priors, h)

    return PriorPredictive(μ, Σ)
end

get_mean(priors::Priors) = priors.m
get_covariance(priors::Priors, h::Hyperparameters) = (priors.R + 1) * (1 - h.β) / (3*h.β - 2) * priors.S