### Batch estimation ###
function estimate_batch!(model::UnivariateStochVolFilter, y::AbstractVector)
    batch = UnivariateStochVolFilter[]

    for t in eachindex(y)
        estimate!(model, y[t])
        push!(batch, deepcopy(model))
    end

    return batch
end

function estimate_batch!(model::MultivariateStochVolFilter, y::AbstractMatrix)
    batch = MultivariateStochVolFilter[]

    for t in axes(y, 1)
        estimate!(model, y[t, :])
        push!(batch, deepcopy(model))
    end

    return batch
end

### Online estimation ###
function estimate!(model::UnivariateStochVolFilter, y::AbstractVector)
    for t in eachindex(y)
        estimate!(model, y[t])   # Update at time t|t and predict at time t+1|t
    end

    return nothing
end

function estimate!(model::MultivariateStochVolFilter, y::AbstractMatrix)
    for t in axes(y, 1)
        estimate!(model, y[t, :])   # Update at time t|t and predict at time t+1|t
    end

    return nothing
end

### Online estimation - single observation ###
function estimate!(model::UnivariateStochVolFilter, y::Real)
    update!(model, y)   # Update at time t|t
    performance!(model) # Calculate performance at time t|t
    predict!(model)     # Predict at time t+1|t

    return nothing
end

function estimate!(model::MultivariateStochVolFilter, y::AbstractVector)
    update!(model, y)   # Update at time t|t
    performance!(model) # Calculate performance at time t|t
    predict!(model)     # Predict at time t+1|t

    return nothing
end

### Update ###
function update!(model::UnivariateStochVolFilter, y::Real)
    model.measurements.y      = y
    model.measurements.errors = model.measurements.y - model.parameters.μ
    model.measurements.scaled = scaled_error(model)

    # Update at time t|t
    Q = model.parameters.P + 1
    K = model.parameters.P / Q
    
    model.parameters.m += K * model.measurements.errors
    model.parameters.P -= K^2 * Q
    model.parameters.S += model.measurements.errors^2/Q

    model.obs += 1

    return nothing
end

function update!(model::MultivariateStochVolFilter, y::AbstractVector)
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

### Predict ###
function predict!(model::AbstractFilter)
    # Predict at time t+1|t
    model.parameters.P = model.parameters.P / model.parameters.hp.δ
    model.parameters.S = model.parameters.S / model.parameters.k

    model.parameters.μ = model.parameters.m
    model.parameters.Σ = (model.parameters.P + 1) * model.parameters.hp.κ * model.parameters.S

    return nothing
end

### Performance ###
function performance!(model::AbstractFilter)
    a = 1/model.obs
    b = 1 - a

    model.performance.LL   = model.performance.LL   * b + a * get_logpdf(model)
    model.performance.ME   = model.performance.ME   * b + a * model.measurements.errors
    model.performance.MAE  = model.performance.MAE  * b + a * abs.(model.measurements.errors)
    model.performance.MSE  = model.performance.MSE  * b + a * model.measurements.errors .^2
    model.performance.MSSE = model.performance.MSSE * b + a * model.measurements.scaled .^2

    return nothing
end

### Log-likelihood
function get_logpdf(model::UnivariateStochVolFilter)
    d = TDist(model.parameters.hp.ν)
    e = (model.measurements.y - model.parameters.μ)/sqrt(model.parameters.Σ)
    l = logpdf(d, e)

    return l
end

function get_logpdf(model::MultivariateStochVolFilter)
    d = MvTDist(model.parameters.hp.ν, model.parameters.μ, model.parameters.Σ)
    l = logpdf(d, model.measurements.y)

    return l
end

### Invert covariance matrix
invert_cholesky(parameters::MultivariateParameters) = inv(cholesky(parameters.Σ))
invert_cholesky(model::MultivariateStochVolFilter)  = inv(cholesky(model.parameters))

# Scaled error
scaled_error(model::AbstractFilter) = scaled_error(model.parameters, model.measurements)

scaled_error(parameters::UnivariateParameters, measurements::UnivariateMeasurements) = measurements.errors / sqrt(parameters.Σ)
scaled_error(parameters::MultivariateParameters, measurements::MultivariateMeasurements) = invert_cholesky(parameters)' * measurements.errors
