
function estimate!(model::MvStochVol, y::AbstractMatrix)
    for t in axes(y, 1)
        estimate!(model, y[t, :])   # Update at time t|t and predict at time t+1|t
    end

    return nothing
end

function estimate!(model::MvStochVol, y::AbstractVector)
    update!(model, y)   # Update at time t|t
    predict!(model)     # Predict at time t+1|t

    return nothing
end

function update!(model::MvStochVol, y::Vector{<:AbstractFloat})
    @assert length(y) == model.p "The measurement vector 'y' must have $(model.p) elements, but has $(length(y))"
    
    model.error = y - model.μ
    model.scaled = invert_cholesky(model) * model.error
    model.loglik = model.loglik + get_logpdf(model, y)

    # Update at time t|t
    Q = model.P + 1
    K = model.P / Q
    
    model.m = model.m + K * model.error
    model.P = model.P - (K * K') * Q
    model.S = model.S + (model.error*model.error')/Q

    model.obs += 1

    return nothing
end

function predict!(model::MvStochVol)
    # Predict at time t+1|t
    model.P = model.P / model.hyperparameters.δ
    model.S = model.S / model.k

    return nothing
end

function get_logpdf(model::MvStochVol, y::Vector{<:Real})
    d = MvTDist(model.hyperparameters.ν, model.μ, model.Σ)
    l = logpdf(d, y)

    return l
end

invert_cholesky(model::MvStochVol) = inv(cholesky(model.Σ))