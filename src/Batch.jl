mutable struct MvStochVolBatch
    measurement::Matrix{<:Real}
    error::Matrix{Float64}
    scaled::Matrix{Float64}
    m::Matrix{Float64}
    P::Vector{Float64}
    S::Array{Float64, 3}
    μ::Matrix{Float64}
    Σ::Array{Float64, 3}
    observations::Integer
    loglikelihood::Float64
    const hyperparameters::Hyperparameters
    const p::Integer
    const k::AbstractFloat

    function MvStochVol(param::Parameters, h::Hyperparameters)
        p = get_p(param)
        y = Vector{Float64}(undef, p)
        e = Vector{Float64}(undef, p)
        z = Vector{Float64}(undef, p)
        k = get_k(h, p)

        # Get prior predictive
        pred = Predictive(prior_μ(param), prior_Σ(param, h))
    
        obs = 0
        ll  = 0.0

        new(y, e, z, param, param, pred, obs, ll, h, p, k)
    end
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

# Extract parameters
get_prior_parameters(model::MvStochVol) = (model.priors.m, model.priors.P, model.priors.S)

function get_prior_parameters(models::Vector{MvStochVol})
    isempty(models) && return nothing

    T = length(models)
    p = models[begin].p
    m = zeros(T, p)
    P = zeros(T)
    S = zeros(T, p, p)

    for t in eachindex(models)
        mt, Pt, St = get_prior_parameters(models[t])

        m[t, :] = mt
        P[t] = Pt
        S[t, :, :] = St
    end

    return (m, P, S)
end