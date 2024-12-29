
function estimate!(model::MvStochVol, y::AbstractMatrix)
    for t in axes(y, 1)
        estimate!(model, y[t, :])
    end

    return nothing
end

function estimate!(model::MvStochVol, y::AbstractVector)
    # Predict at time t|t-1
    predict!(model)
    
    # Update at time t|t
    update!(model, y)

    return nothing
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
    dist = MvTDist(model.hyperparameters.ν, model.predictive.μ, model.predictive.Σ)
    model.loglikelihood += logpdf(dist, y)

    model.observations += 1
    
    return nothing
end

#function prior_predictive!(model::MvStochVol)
#    model.predictive.μ = prior_μ(model.parameters.m)
#    model.predictive.Σ = prior_Σ(model.parameters.P, model.parameters.S, model.hyperparameters)

#    return nothing
#end

#μ = m
    #Σ = (P + 1) * (1 - h.β) / (2*h.β - 1) * S

#error(y::FLOATVEC, μ::FLOATVEC) = y - μ
#error(y::FLOATVEC, pred::PriorPredictive) = y - pred.μ

invert_cholesky(Σ::AbstractMatrix) = inv(cholesky(Σ).L)

#standardised_error(e, Σ::AbstractMatrix)              = invert_cholesky(Σ) * e
#standardised_error(e, pred::PriorPredictive) = standardised_error(pred.Σ) * e