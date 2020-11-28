
"""
simulate!(model::UnivariateModel)

Simulate a univariate stochastic volatility model given the priors in ``model` with ``model.y`` updated in place.
"""

function simulate!(model::UnivariateModel)
    T = length(model.y)
    m = model.m
    P = model.P
    S = model.S
    for t in 1:T
        model.y[t] = rand(Normal(m, S))
        m = rand(Normal(m, S * P))
    end
end

"""
simulate!(model::MultivariateModel)

Simulate a multivariate stochastic volatility model given the priors in ``model` with ``model.y`` updated in place.
"""

function simulate!(model::MultivariateModel)
    T, p = size(model.y)
    m = model.m
    P = model.P
    S = model.S
    for t in 1:T
        model.y[t, :] = rand(MvNormal(m, S))
        m = rand(MvNormal(m, S .* P))
    end
end