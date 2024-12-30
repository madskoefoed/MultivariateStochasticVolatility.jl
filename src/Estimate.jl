
function estimate!(model::MvStochVol, y::AbstractMatrix)
    for t in axes(y, 1)
        estimate!(model, y[t, :])   # Predict at time t|t-1 and update at time t|t
    end

    return nothing
end

function estimate!(model::MvStochVol, y::AbstractVector)
    predict!(model)     # Predict at time t|t-1
    update!(model, y)   # Update at time t|t

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
    model.priors.P = model.posteriors.P/model.hyperparameters.δ
    model.priors.S = model.posteriors.S/model.k

    model.predictive.μ = prior_μ(model.priors)
    model.predictive.Σ = prior_Σ(model.priors, model.hyperparameters)

    return nothing
end

function update!(model::MvStochVol, y::Vector{<:AbstractFloat})
    @assert length(y) == model.p "The measurement vector 'y' must have $(model.p) elements, but has $(length(y))"

    model.measurement = y

    # Prediction error
    model.error = model.measurement - model.priors.m

    # Scaled prediction error
    model.scaled = invert_cholesky(model.predictive.Σ) * model.error

    # Update
    Q = model.priors.P + 1
    K = model.priors.P / Q
    
    model.posteriors.m = model.priors.m + K * model.error
    model.posteriors.P = model.priors.P - (K * K') * Q
    model.posteriors.S = model.priors.S + (model.error*model.error')/Q
    
    # Predictive log-likelihood
    dist = prior_distribution(model)
    model.loglikelihood += logpdf(dist, y)

    model.observations += 1
    
    return nothing
end

invert_cholesky(Σ::AbstractMatrix) = inv(cholesky(Σ).L)