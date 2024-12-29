
function update!(model::MvStochVol, y::AbstractMatrix)
    for t in axes(y, 1)
        update!(model, y[t, :])
    end

    return nothing
end

function update!(model::MvStochVol, y::AbstractVector)
    model.measurement = y

    # Update at time t|t
    posterior_parameters!(model, y)

    # Predict at time t|t-1
    prior_parameters!(model)
    prior_predictive!(model)

    # Predictive log-likelihood
    dist = MvTDist(model.hyperparameters.ν, model.predictive.μ, model.predictive.Σ)
    llt = logpdf(dist, y)
    model.loglikelihood += llt

    model.observations += 1

    return nothing
end

function prior_parameters!(model::MvStochVol)
    model.parameters.m = model.parameters.m
    model.parameters.P = model.parameters.P/model.hyperparameters.δ
    model.parameters.S = model.parameters.S/model.k

    return nothing
end

function posterior_parameters!(model::MvStochVol, y::Vector{<:AbstractFloat})
    # Prediction error
    model.error = y - model.parameters.m

    # Update
    Q = model.parameters.P + 1
    K = model.parameters.P / Q
    
    model.parameters.m += K * model.error
    model.parameters.P -= K * K' * Q
    model.parameters.S += model.error*model.error'/Q
    
    return nothing
end

function prior_predictive!(model::MvStochVol)
    model.predictive.μ = prior_μ(model.parameters.m)
    model.predictive.Σ = prior_Σ(model.parameters.P, model.parameters.S, model.hyperparameters)

    return nothing
end

#μ = m
    #Σ = (P + 1) * (1 - h.β) / (2*h.β - 1) * S

#error(y::FLOATVEC, μ::FLOATVEC) = y - μ
#error(y::FLOATVEC, pred::PriorPredictive) = y - pred.μ

#inert_cholesky(Σ::AbstractMatrix) = inv(cholesky(Σ).L)

#standardised_error(e, Σ::AbstractMatrix)              = invert_cholesky(Σ) * e
#standardised_error(e, pred::PriorPredictive) = standardised_error(pred.Σ) * e