
function estimate!(model::MvStochVol, y::AbstractMatrix)
    for t in axes(y, 1)
        estimate!(model, y[t, :])   # Predict at time t|t-1 and update at time t|t
    end

    return nothing
end

function estimate!(model::MvStochVol, y::AbstractVector)
    update!(model, y)   # Update at time t|t
    predict!(model)     # Predict at time t+1|t

    return nothing
end

function estimate(model::MvStochVol, y::AbstractMatrix)
    T = size(y, 1)
    models = Vector{MvStochVol}(undef, T)
    for t in axes(y, 1)
        estimate!(model, y[t, :])

        models[t] = model
    end

    return models
end

function predict!(model::MvStochVol)
    model.priors.m = model.posteriors.m
    model.priors.R = model.posteriors.P/model.hyperparameters.δ
    model.priors.S = model.posteriors.S/model.k

    model.prior_predictive.μ = get_mean(model.priors)
    model.prior_predictive.Σ = get_covariance(model.priors, model.hyperparameters)

    return nothing
end

function update!(model::MvStochVol, y::Vector{<:AbstractFloat})
    @assert length(y) == model.p "The measurement vector 'y' must have $(model.p) elements, but has $(length(y))"

    model.measurements = y

    # Prediction error
    model.errors = model.measurements - model.priors.m

    # Scaled prediction error
    model.standardized = invert_cholesky(model.prior_predictive.Σ) * model.errors

    # Update
    Q = model.priors.R + 1
    K = model.priors.R / Q
    
    model.posteriors.m = model.priors.m + K * model.errors
    model.posteriors.P = model.priors.R - (K * K') * Q
    model.posteriors.S = model.priors.S + (model.errors*model.errors')/Q
    
    # Predictive log-likelihood
    #dist = prior_distribution(model)
    #model.loglikelihood += logpdf(dist, y)

    model.observations += 1
    
    return nothing
end

invert_cholesky(Σ::AbstractMatrix) = inv(cholesky(Σ).L)