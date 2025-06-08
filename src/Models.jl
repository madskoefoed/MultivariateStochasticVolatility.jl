"""
    AbstractFilter

Abstract type representing a filter
"""
abstract type AbstractFilter end

"""
    Filter

A mutable struct, Filter, which contains the updated filter information at time t.
The filter is initialized at time t=0 by supplying an `AbstractParameters` struct.

# Members

- `obs`: number of updates (time steps) performed up to and including time t
- `parameters`: a struct with parameters
- `μ`: vector of mean predictions
- `Σ`: a covariance matrix of predictions
- `y`: a vector of measurements at time t
- `errors`: a vector of errors at time t
- `scaled`: a vector of standardized errors at time t
- `performance`: a mutable struct, performance, with model performance data
"""
mutable struct Filter <: AbstractFilter
    obs::Integer
    parameters::AbstractParameters
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    y::Vector{Float64}
    errors::Vector{Float64}
    scaled::Vector{Float64}
    performance::FilterPerformance
    
    function Filter(param::PARAM) where {PARAM <: AbstractParameters}

        p = size(param.S, 1)

        m, P, S, μ, Σ = predict(param)

        perf = FilterPerformance(p)

        print(typeof(Σ))

        new(0, param, μ, Σ, zeros(p), zeros(p), zeros(p), perf)
    end
end