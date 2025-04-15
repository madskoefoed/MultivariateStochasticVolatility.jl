abstract type AbstractFilter end

mutable struct UnivariateStochVolFilter <: AbstractFilter
    obs::Integer
    priors::UnivariatePriors
    parameters::UnivariateParameters
    measurements::UnivariateMeasurements
    performance::UnivariatePerformance
    
    function UnivariateStochVolFilter(priors::UnivariatePriors, hp::Hyperparameters)
        p = size(priors.S, 1)
        param = MultivariateStochasticVolatility.UnivariateParameters(priors, hp);

        new(0, priors, param, UnivariateMeasurements(), UnivariatePerformance())
    end
end

mutable struct MultivariateStochVolFilter <: AbstractFilter
    obs::Integer
    priors::MultivariatePriors
    parameters::MultivariateParameters
    measurements::MultivariateMeasurements
    performance::MultivariatePerformance
    
    function MultivariateStochVolFilter(priors::MultivariatePriors, hp::Hyperparameters)
        p = size(priors.S, 1)
        param = MultivariateStochasticVolatility.MultivariateParameters(priors, hp);

        new(0, priors, param, MultivariateMeasurements(p), MultivariatePerformance(p))
    end
end

##########################
### Outer constructors ###
##########################

StochVolFilter(priors::UnivariatePriors, hp::Hyperparameters)   = UnivariateStochVolFilter(priors, hp)
StochVolFilter(priors::MultivariatePriors, hp::Hyperparameters) = MultivariateStochVolFilter(priors, hp)