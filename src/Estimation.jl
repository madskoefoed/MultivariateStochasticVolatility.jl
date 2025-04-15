function estimate_batch!(model::MvStochVolFilter, y::AbstractMatrix)
    batch = MvStochVolFilter[]

    for t in axes(y, 1)
        estimate!(model, y[t, :])
        push!(batch, deepcopy(model))
    end

    return batch
end


function estimate!(model::MvStochVolFilter, y::AbstractMatrix)
    for t in axes(y, 1)
        estimate!(model, y[t, :])   # Update at time t|t and predict at time t+1|t
    end

    return nothing
end

function estimate!(model::MvStochVolFilter, y::AbstractVector)
    update!(model, y)   # Update at time t|t
    predict!(model)     # Predict at time t+1|t
    performance!(model)

    return nothing
end

function update!(model::MvStochVolFilter, y::AbstractVector)
    @assert length(y) == model.parameters.p "The measurement vector 'y' must have $(model.parameters.p) elements, but has $(length(y))"
    
    model.measurements.y      = y
    model.measurements.errors = model.measurements.y - model.parameters.μ
    model.measurements.scaled = scaled_error(model)
    #model.measurements.scaled = model.measurements.errors ./ sqrt.(diag(model.parameters.Σ))

    # Update at time t|t
    Q = model.parameters.P + 1
    K = model.parameters.P / Q
    
    model.parameters.m += K * model.measurements.errors
    model.parameters.P -= K^2 * Q #(K * K') * Q
    model.parameters.S += (model.measurements.errors * model.measurements.errors')/Q

    model.obs += 1

    return nothing
end

function predict!(model::MvStochVolFilter)
    # Predict at time t+1|t
    model.parameters.P = model.parameters.P / model.parameters.hp.δ
    model.parameters.S = model.parameters.S / model.parameters.k

    model.parameters.μ = model.parameters.m
    model.parameters.Σ = (model.parameters.P + 1) * model.parameters.hp.κ * model.parameters.S

    return nothing
end

function performance!(model::MvStochVolFilter)
    a = 1/model.obs
    b = 1 - a

    model.performance.LL   = model.performance.LL   * b + a * get_logpdf(model)
    model.performance.ME   = model.performance.ME   * b + a * model.measurements.errors
    model.performance.MAE  = model.performance.MAE  * b + a * abs.(model.measurements.errors)
    model.performance.MSE  = model.performance.MSE  * b + a * model.measurements.errors .^2
    model.performance.MSSE = model.performance.MSSE * b + a * model.measurements.scaled .^2

    return nothing
end

function get_logpdf(model::MvStochVolFilter)
    d = MvTDist(model.parameters.hp.ν, model.parameters.μ, model.parameters.Σ)
    l = logpdf(d, model.measurements.y)

    return l
end

invert_cholesky(parameters::Parameters)  = inv(cholesky(parameters.Σ))
invert_cholesky(model::MvStochVolFilter) = inv(cholesky(model.parameters))

scaled_error(parameters::Parameters, measurements::Measurements) = invert_cholesky(parameters)' * measurements.errors
scaled_error(model::MvStochVolFilter) = scaled_error(model.parameters, model.measurements)