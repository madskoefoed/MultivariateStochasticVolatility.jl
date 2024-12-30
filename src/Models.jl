mutable struct MvStochVol
    measurements::Vector{<:Real}
    errors::Vector{Float64}
    standardized::Vector{Float64}
    priors::Priors
    posteriors::Posteriors
    prior_predictive::PriorPredictive
    #posterior_predictive::PosteriorPredictive
    observations::Integer
    loglikelihood::Float64
    const hyperparameters::Hyperparameters
    const p::Integer
    const k::AbstractFloat

    function MvStochVol(priors::Priors, h::Hyperparameters)
        
        posteriors = Posteriors(priors.m, priors.R, priors.S)

        p = get_p(priors)
        y = Vector{Float64}(undef, p)
        e = Vector{Float64}(undef, p)
        z = Vector{Float64}(undef, p)
        k = get_k(h, p)

        # Get prior predictive
        pred = PriorPredictive(priors, h)
    
        obs = 0
        ll  = 0.0

        new(y, e, z, priors, posteriors, pred, obs, ll, h, p, k)
    end
end

#get_loglik(model::MvStochVol) = logpdf(prior_distribution(model), model.measurement)

#function prior_distribution(model::MvStochVol)
#    return MvTDist(model.hyperparameters.ν, model.predictive.μ, model.predictive.Σ)
#end