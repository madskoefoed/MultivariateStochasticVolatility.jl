
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
    lp = length(model.predictions)
    lm = length(model.updates)
    @assert lp == lm + 1 "There must be $(lm+1) prediction steps, but there are $lp"

    @assert length(y) == model.p "The measurement vector 'y' must have $(model.p) elements, but has $(length(y))"
    
    update_step = UpdateStep(model.predictions[end], model.hyperparameters, y)

    push!(model.updates, update_step)

    return nothing
end

function predict!(model::MvStochVol)
    lp = length(model.predictions)
    lm = length(model.updates)
    @assert lp == lm "To be able to make a prediction at time t+1, the model needs to be updated with information at time t"

    predict_step = PredictStep(model.updates[end], model.hyperparameters)

    push!(model.predictions, predict_step)

    return nothing
end