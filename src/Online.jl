mutable struct Measurements
    y::Vector{<:Real}
    errors::Vector{Float64}
    scaled::Vector{Float64}
end

Measurements(p::Integer) = Measurements(zeros(p), zeros(p), zeros(p))

mutable struct MvStochVolOnline <: MvStochVol
    parameters::Parameters
    measurements::Measurements
    loglik::Float64
    obs::Integer

    function MvStochVolOnline(parameters::Parameters)
        measurements = Measurements(parameters.p)

        new(parameters, measurements, 0.0, 0)
    end
end

function estimate!(model::MvStochVolOnline, y::AbstractMatrix)
    for t in axes(y, 1)
        estimate!(model, y[t, :])   # Update at time t|t and predict at time t+1|t
    end

    return nothing
end

function estimate!(model::MvStochVolOnline, y::AbstractVector)
    update!(model, y)   # Update at time t|t
    predict!(model)     # Predict at time t+1|t

    return nothing
end

function update!(model::MvStochVolOnline, y::AbstractVector)
    @assert length(y) == model.parameters.p "The measurement vector 'y' must have $(model.parameters.p) elements, but has $(length(y))"
    
    model.measurements.y = y
    model.measurements.errors = y - model.parameters.μ
    model.measurements.scaled = invert_cholesky(model) * model.measurements.errors
    model.loglik = model.loglik + get_logpdf(model)

    # Update at time t|t
    Q = model.parameters.P + 1
    K = model.parameters.P / Q
    
    model.parameters.m = model.parameters.m + K * model.measurements.errors
    model.parameters.P = model.parameters.P - (K * K') * Q
    model.parameters.S = model.parameters.S + (model.measurements.errors*model.measurements.errors')/Q

    model.obs += 1

    return nothing
end

function predict!(model::MvStochVolOnline)
    # Predict at time t+1|t
    model.parameters.P = model.parameters.P / model.parameters.hyper.δ
    model.parameters.S = model.parameters.S / model.parameters.k

    model.parameters.μ = model.parameters.m
    model.parameters.Σ = (model.parameters.P + 1) * (1 - model.parameters.hyper.β) / (3*model.parameters.hyper.β - 2) * model.parameters.S


    return nothing
end

function get_logpdf(model::MvStochVolOnline)
    d = MvTDist(model.parameters.hyper.ν, model.parameters.μ, model.parameters.Σ)
    l = logpdf(d, model.measurements.y)

    return l
end

invert_cholesky(parameters::Parameters)  = inv(cholesky(parameters.Σ))
invert_cholesky(model::MvStochVolOnline) = inv(cholesky(model.parameters.Σ))