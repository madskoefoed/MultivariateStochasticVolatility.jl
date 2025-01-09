abstract type TimeStep end

struct PredictStep <: TimeStep
    parameters::Parameters
    μ::Vector{Float64}
    Σ::Matrix{Float64}
end

struct UpdateStep <: TimeStep
    parameters::Parameters
    y::Vector{<:Real}
    error::Vector{Float64}
    scaled::Vector{Float64}
    logpdf::Float64
end

##########################
### Outer constructors ###
##########################

function PredictStep(param::Parameters, hyper::Hyperparameters)
    μ = get_prior_mean(param)
    Σ = get_prior_covariance(param, hyper)

    return PredictStep(param, μ, Σ)
end

function PredictStep(timestep::UpdateStep, hyper::Hyperparameters)
    # Predict at time t+1|t
    p = get_p(timestep)
    k = get_k(hyper, p)
    P = timestep.parameters.P / hyper.δ
    S = timestep.parameters.S/k

    return PredictStep(Parameters(timestep.parameters.m, P, S), hyper)
end

function UpdateStep(timestep::PredictStep, hyper::Hyperparameters, y::Vector{<:Real})
    e = y - timestep.μ
    z = invert_cholesky(timestep.Σ) * e
    l = get_logpdf(timestep, hyper, y)

    # Update at time t|t
    Q = timestep.parameters.P + 1
    K = timestep.parameters.P / Q
    
    m = timestep.parameters.m + K * e
    P = timestep.parameters.P - (K * K') * Q
    S = timestep.parameters.S + (e*e')/Q
    
    return UpdateStep(Parameters(m, P, S), y, e, z, l)
end

#################
### Functions ###
#################

get_p(timestep::TimeStep) = size(timestep.parameters.S, 1)

invert_cholesky(Σ::AbstractMatrix) = inv(cholesky(Σ).L)

get_prior_mean(param::Parameters) = param.m
get_prior_covariance(param::Parameters, hyper::Hyperparameters) = (param.P + 1) * (1 - hyper.β) / (3*hyper.β - 2) * param.S

get_posterior_mean(param::Parameters) = param.m
get_posterior_covariance(param::Parameters, hyper::Hyperparameters) = (param.P + 1) * (1 - hyper.β) / (2*hyper.β - 1) * param.S

get_distribution(step::PredictStep, hyper::Hyperparameters) = MvTDist(hyper.ν, step.μ, step.Σ)

get_logpdf(step::PredictStep, hyper::Hyperparameters, y::Vector{<:Real}) = logpdf(get_distribution(step, hyper), y)