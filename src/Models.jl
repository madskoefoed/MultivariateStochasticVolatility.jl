mutable struct MvStochVol
    obs::Integer
    parameters::Parameters
    measurements::Measurements
    performance::Performance
    
    MvStochVol(parameters::Parameters) = new(0, parameters, Measurements(parameters.p), Performance(parameters.p))
end
