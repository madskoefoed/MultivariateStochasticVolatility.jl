abstract type AbstractStochVolFilter end

mutable struct MvStochVolFilter <: AbstractStochVolFilter
    obs::Integer
    parameters::Parameters
    measurements::Measurements
    performance::Performance
    
    MvStochVolFilter(parameters::Parameters) = new(0, parameters, Measurements(parameters.p), Performance(parameters.p))
end
