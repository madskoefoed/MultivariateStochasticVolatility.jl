### Batch estimation ###
function estimate_batch!(model::Filter, y::AbstractMatrix)
    batch = Filter[]

    for t in axes(y, 1)
        estimate!(model, y[t, :])
        push!(batch, deepcopy(model))
    end

    return batch
end

### Online estimation ###
function estimate!(model::Filter, y::AbstractMatrix)
    for t in axes(y, 1)
        estimate!(model, y[t, :])   # Update at time t|t and predict at time t+1|t
    end

    return nothing
end

### Online estimation - single observation ###
function estimate!(model::Filter, y::AbstractVector)
    update!(model, y)   # Update at time t|t
    performance!(model) # Calculate performance at time t|t
    predict!(model)     # Predict at time t+1|t

    return nothing
end

### Update ###
function update!(model::Filter, y::AbstractVector)
    @assert length(y) == model.parameters.p "The measurement vector 'y' must have $(model.parameters.p) elements, but has $(length(y))"
    
    model.y      = y
    model.errors = model.y - model.μ
    model.scaled = scaled_error(model)

    # Update at time t|t
    Q = model.parameters.P + 1
    K = model.parameters.P / Q
    
    model.parameters.m += K * model.errors
    model.parameters.P -= (K * K') * Q
    model.parameters.S += (model.errors * model.errors') / Q

    model.obs += 1

    return nothing
end

### Predict ###
function predict!(model::Filter)
    # Predict at time t+1|t
    model.parameters.P = model.parameters.P / model.parameters.hyper.δ
    model.parameters.S = model.parameters.S / model.parameters.k

    model.μ = model.parameters.m
    model.Σ = (model.parameters.P + 1) * model.parameters.hyper.κ * model.parameters.S

    return nothing
end

function predict(param::MeanParameters)
    # Predict at time t+1|t
    m = deepcopy(param.m)
    P = param.P / param.hyper.δ
    S = param.S / param.k

    μ = param.m
    Σ = (P + 1) * param.hyper.κ * S

    return m, P, S, μ, Σ
end

### Performance ###
function performance!(model::AbstractFilter)
    a = 1/model.obs
    b = 1 - a

    model.performance.LL   = model.performance.LL   * b + a * get_logpdf(model)
    model.performance.ME   = model.performance.ME   * b + a * model.errors
    model.performance.MAE  = model.performance.MAE  * b + a * abs.(model.errors)
    model.performance.MSE  = model.performance.MSE  * b + a * model.errors .^2
    model.performance.MSSE = model.performance.MSSE * b + a * model.scaled .^2

    return nothing
end

### Log-likelihood
function get_logpdf(model::Filter)
    d = MvTDist(model.parameters.hyper.ν, model.μ, model.Σ)
    l = logpdf(d, model.y)

    return l
end

### Invert covariance matrix
invert_cholesky(model::Filter) = inv(cholesky(model.Σ))

# Scaled error
scaled_error(model::AbstractFilter) = invert_cholesky(model)' * model.errors
