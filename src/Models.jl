mutable struct MvStochVol
    parameters::Parameters
    measurements::Measurements
    loglik::Float64
    obs::Integer

    function MvStochVol(parameters::Parameters)
        measurements = Measurements(parameters.p)

        new(parameters, measurements, 0.0, 0)
    end
end

mutable struct MvStochVolHistory
    online::Vector{MvStochVol}
    const history::Integer

    function MvStochVolHistory(parameters::Parameters, history::Integer)
        @assert history >= 0 "The integer $history must be positive, but is $history < 0"

        online = MvStochVol(parameters)
        online = MvStochVol[online]

        new(online, history)
    end
end