"""
    batch!(model::Filter, y::AbstractMatrix)

Estimate a filter model to all the data in batch mode where `y` is expected to be a multivariate variable
with observations along the first axis. Batch mode implies that `model` is updated for t=1,...,T while
a vector `Vector{Filter}` is returned.
"""
function batch!(model::Filter, y::AbstractMatrix{<:Real})
    batch = Filter[]

    for t in axes(y, 1)
        estimate!(model, y[t, :])
        push!(batch, deepcopy(model))
    end

    return batch
end

"""
    estimate!(model::Filter, y::AbstractVector)
    estimate!(model::Filter, y::AbstractMatrix)
    
Estimate a filter model for either a single data point or multiple data points.

In the first method, `y` corresponds to a single observation.
In the second method, `y` corresponds to multiple observations.
"""
function estimate!(model::Filter, y::AbstractVector{<:Real})
    update!(model, y)   # Update at time t|t
    performance!(model) # Calculate performance at time t|t
    predict!(model)     # Predict at time t+1|t

    return nothing
end

function estimate!(model::Filter, y::AbstractMatrix{<:Real})
    for t in axes(y, 1)
        estimate!(model, y[t, :])   # Update at time t|t and predict at time t+1|t
    end

    return nothing
end

### Update ###
function update!(model::Filter, y::AbstractVector{<:Real})
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

function predict(param::Parameters)
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
