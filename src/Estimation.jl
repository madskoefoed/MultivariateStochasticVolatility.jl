function estimate(param::AbstractParameters, y::AbstractMatrix)
    for t in axes(y, 1)
        model = estimate(param, y[t, :])
    end

    return model
end

estimate(model::FilterModel, y::AbstractVector) = estimate(model.parameters, y)
function estimate(param::AbstractParameters, y::AbstractVector)
    @assert length(y) == get_p(param) "The measurement vector 'y' must have $(param.p) elements, but has $(length(y))"

    e = convert(y, Vector{Float64}) - param.μ
    z = invert_cholesky(param) * e

    measure = Measurements(y, e, z)

    #perf = Performance(model.parameters, measure, model.performance)

    # Update at time t|t
    Q = param.P + 1
    K = param.P / Q
    
    m = param.m + K * e
    P = param.P - (K * K') * Q
    S = param.S + (e * e')/Q

    # Predict at time t+1|t
    param = Parameters(m, P, S, param.hyper)

    return param, measure
end

function performance!(model::FilterModel)
    a = 1/model.obs
    b = 1 - a

    #model.performance.loglikelihood = model.performance.loglikelihood * b + get_logpdf(model) * a
    model.performance.LL += get_logpdf(model)

    model.performance.ME   = model.performance.ME * b + a * model.measurements.errors
    model.performance.MAE  = model.performance.MAE * b + a * abs.(model.measurements.errors)
    model.performance.MSE  = model.performance.MSE * b + a * model.measurements.errors .^2
    model.performance.MSSE = model.performance.MSSE * b + a * model.measurements.scaled .^2

    return nothing
end

function get_logpdf(model::FilterModel)
    d = MvTDist(param.hyper.ν, param.μ, param.Σ)
    l = logpdf(d, model.measurements.y)

    return l
end

invert_cholesky(parameters::Parameters) = inv(cholesky(parameters.Σ))
invert_cholesky(model::MvStochVol)      = inv(cholesky(param.Σ))