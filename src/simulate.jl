
"""
simulate!(model::MultivariateModel)

Simulate a stochastic volatility model given m, P, and S. If P > 0, the mean vector is simulated
as a random walk while P = 0 implies a standard stochastic volatility model without time-varying mean(s).
"""

function simulate(TVVAR::UnivariateModel)
    x = TVVAR.x
    m = TVVAR.m
    P = TVVAR.P
    S = TVVAR.S
    for t in 1:lenth(x)
            TVVAR.x[t] = rand(Normal(TVVAR.m, TVVAR.S))
            TVVAR.m = rand(Normal(m[1], S[1] .* P))
    end
    return y
end

function simulate!(TVVAR::MultivariateModel)
    T, p = size(TVVAR.x)
    for t in 1:T
        if p == 1
            y[t, :] = rand(Normal(m[1], S[1]))
            m[1] = rand(Normal(m[1], S[1] .* P))
        else
            y[t, :] = rand(MvNormal(m, S))
            m = rand(MvNormal(m, S .* P))
        end
    end
    return y
end