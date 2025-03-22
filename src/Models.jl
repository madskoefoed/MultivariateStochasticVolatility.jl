mutable struct MvStochVol
    parameters::Parameters
    measurements::Measurements
    loglik::Float64
    obs::Integer

    MvStochVol(parameters::Parameters) = new(parameters, Measurements(parameters.p), 0.0, 0)
end