abstract type AbstractStochVolFilter end

mutable struct MvStochVolFilter <: AbstractStochVolFilter
    obs::Integer
    priors::Priors
    parameters::Parameters
    measurements::Measurements
    performance::Performance
    
    function MvStochVolFilter(priors::Priors, hp::Hyperparameters)
        param = MultivariateStochasticVolatility.Parameters(priors, hp);

        new(0, priors, param, Measurements(param.p), Performance(param.p))
    end
end
