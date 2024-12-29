mutable struct MvStochVol
    measurement::Vector{<:Real}
    error::Vector{Float64}
    scaled::Vector{Float64}
    priors::Parameters
    posteriors::Parameters
    predictive::Predictive
    observations::Integer
    loglikelihood::Float64
    const hyperparameters::Hyperparameters
    const p::Integer
    const k::AbstractFloat

    function MvStochVol(param::Parameters, h::Hyperparameters)
        p = get_p(param)
        y = Vector{Float64}(undef, p)
        e = Vector{Float64}(undef, p)
        z = Vector{Float64}(undef, p)
        k = get_k(h, p)

        # Get prior predictive
        pred = Predictive(prior_μ(param), prior_Σ(param, h))
    
        obs = 0
        ll  = 0.0

        new(y, e, z, param, param, pred, obs, ll, h, p, k)
    end
end

get_p(param::Parameters)              = size(param.S, 1)
get_k(h::Hyperparameters, p::Integer) = (h.β - p*h.β + p)/(h.β - p*h.β + p - 1)
get_df(h::Hyperparameters)            = h.ν 
get_loglik(model::MvStochVol)         = logpdf(MvTDist(model.hyperparameters.ν, model.predictive.μ, model.predictive.Σ),
                                               model.measurement)

function prior_distribution(model::MvStochVol)
    return MvTDist(model.hyperparameters.ν, model.predictive.μ, model.predictive.Σ)
end