get_parameters(model::AbstractFilter) = (model.parameters.m, model.parameters.P, model.parameters.S)

function get_parameters(model::Vector{Filter})
    @assert !isempty(model) "The vector 'model' cannot be empty."

    T = length(model)
    p = model[1].parameters.p
    m = zeros(T, p)
    P = zeros(T)
    S = zeros(T, p, p)
    for t in eachindex(model)
        m[t, :]    = model[t].parameters.m
        P[t]       = model[t].parameters.P
        S[t, :, :] = model[t].parameters.S
    end

    return (m, P, S)
end

get_predictions(model::AbstractFilter) = (model.μ, model.Σ)

function get_predictions(model::Vector{Filter})
    @assert !isempty(model) "The vector 'model' cannot be empty."

    T = length(model)
    p = model[1].parameters.p
    μ = zeros(T, p)
    Σ = zeros(T, p, p)
    for t in eachindex(model)
        μ[t, :]    = model[t].μ
        Σ[t, :, :] = model[t].Σ
    end

    return (μ, Σ)
end