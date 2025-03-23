function estimate!(model::MvStochVol, y::AbstractMatrix)
    for t in axes(y, 1)
        estimate!(model, y[t, :])   # Update at time t|t and predict at time t+1|t
    end

    return nothing
end

function estimate!(model::MvStochVol, y::AbstractVector)
    update!(model, y)   # Update at time t|t
    predict!(model)     # Predict at time t+1|t
    performance!(model)

    return nothing
end

function update!(model::MvStochVol, y::AbstractVector)
    @assert length(y) == model.parameters.p "The measurement vector 'y' must have $(model.parameters.p) elements, but has $(length(y))"
    
    model.measurements.y = y
    model.measurements.errors = y - model.parameters.μ
    model.measurements.scaled = invert_cholesky(model) * model.measurements.errors

    # Update at time t|t
    Q = model.parameters.P + 1
    K = model.parameters.P / Q
    
    model.parameters.m = model.parameters.m + K * model.measurements.errors
    model.parameters.P = model.parameters.P - (K * K') * Q
    model.parameters.S = model.parameters.S + (model.measurements.errors*model.measurements.errors')/Q

    model.obs += 1

    return nothing
end

function predict!(model::MvStochVol)
    # Predict at time t+1|t
    model.parameters.P = model.parameters.P / model.parameters.hyper.δ
    model.parameters.S = model.parameters.S / model.parameters.k

    model.parameters.μ = model.parameters.m
    model.parameters.Σ = (model.parameters.P + 1) * (1 - model.parameters.hyper.β) / (3*model.parameters.hyper.β - 2) * model.parameters.S


    return nothing
end

function performance!(model::MvStochVol)
    a = 1/model.obs
    b = 1 - a

    #model.performance.loglikelihood = model.performance.loglikelihood * b + get_logpdf(model) * a
    model.performance.loglikelihood += get_logpdf(model)

    model.performance.mean_error = model.performance.mean_error * b + a * model.measurements.errors
    model.performance.mean_absolute_error = model.performance.mean_absolute_error * b + a * abs.(model.measurements.errors)
    model.performance.mean_squared_error = model.performance.mean_squared_error * b + a * model.measurements.errors .^2
    model.performance.mean_squared_standardized_error = model.performance.mean_squared_standardized_error * b + a * model.measurements.scaled .^2

    return nothing
end

function get_logpdf(model::MvStochVol)
    d = MvTDist(model.parameters.hyper.ν, model.parameters.μ, model.parameters.Σ)
    l = logpdf(d, model.measurements.y)

    return l
end

invert_cholesky(parameters::Parameters) = inv(cholesky(parameters.Σ))
invert_cholesky(model::MvStochVol)      = inv(cholesky(model.parameters.Σ))