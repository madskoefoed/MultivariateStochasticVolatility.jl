mutable struct MvStochVol
    measurement::Vector{<:Real}
    error::Vector{<:Real}
    parameters::Parameters
    predictive::Predictive
    #loglikelihood::AbstractFloat
    const hyperparameters::Hyperparameters
    const p::Integer
    const k::AbstractFloat

    function MvStochVol(param::Parameters, h::Hyperparameters)
        p = get_p(param)
        y = Vector{Float64}(undef, p)
        e = Vector{Float64}(undef, p)
        k = get_k(h, p)

        # Get prior predictive
        pred = get_predictive(param, h)

        #ll  = 0.0

        new(y, e, param, pred, h, p, k)
    end
end

get_p(param::Parameters) = size(param.S, 1)
get_k(h::Hyperparameters, p::Integer) = (h.β - p*h.β + p)/(h.β - p*h.β + p - 1)
get_df(h::Hyperparameters) = h.ν 

get_loglik(model::MvStochVol) = logpdf(MvTDist(model.hyperparameters.ν, model.predictive.μ, model.predictive.Σ), model.measurement)