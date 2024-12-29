mutable struct Predictive
    μ::Vector{<:AbstractFloat}
    Σ::AbstractMatrix

    function Predictive(μ::Vector{<:AbstractFloat}, Σ::AbstractMatrix)

        str = "μ is a $(length(μ))-dimensional vector, but Σ is a $(size(Σ, 1)) x $(size(Σ, 2)) matrix."
        !(length(μ) == size(Σ, 1) == size(Σ, 2)) && throw(DimensionMismatch(str))
    
        new(μ, Σ)
    end
end

function prior_predictive(param::Parameters, h::Hyperparameters)
    μ = prior_μ(param.m)
    Σ = prior_Σ(param.P, param.S, h)

    return Predictive(μ, Σ)
end

prior_μ(m::Vector{<:AbstractFloat}) = m
prior_Σ(P::AbstractFloat, S::AbstractMatrix, h::Hyperparameters) = (P + 1) * (1 - h.β) / (3*h.β - 2) * S

prior_μ(param::Parameters) = param.m
prior_Σ(param::Parameters, h::Hyperparameters) = prior_Σ(param.P, param.S, h)