abstract type AbstractFilter end

mutable struct Filter <: AbstractFilter
    obs::Integer
    parameters::AbstractParameters
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    y::Vector{Float64}
    errors::Vector{Float64}
    scaled::Vector{Float64}
    performance::Performance
    
    function Filter(param::PARAM) where {PARAM <: AbstractParameters}

        p = size(param.S, 1)

        m, P, S, μ, Σ = predict(param)

        perf = Performance(p)

        print(typeof(Σ))

        new(0, param, μ, Σ, zeros(p), zeros(p), zeros(p), perf)
    end
end

##########################
### Outer constructors ###
##########################

#Filter(priors::MultivariatePriors, hyper::Hyperparameters) = MultivariateFilter(priors, hyper)