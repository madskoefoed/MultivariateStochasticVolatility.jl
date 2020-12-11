
"""
simulate!(ss::StateSpace)

Simulate a stochastic volatility model given m, P, and S. If P > 0, the mean vector is simulated
as a random walk while P = 0 implies a standard stochastic volatility model without time-varying mean(s).
"""

function simulate!(ssm::StateSpace)
    y = ssm.y
    x = ssm.x
    # Constants
    T, J = size(y)
    D    = size(x, 2)
    # Storage
    m = zeros(T + 1, D, J)
    P = zeros(T + 1, D, D)
    S = zeros(T + 1, J, J)

    m[1, :, :] = ssm.priors.m
    P[1, :, :] = ssm.priors.P
    S[1, :, :] = ssm.priors.S

    for t = 1:T
        y[t, :]        = rand(Distributions.MvNormal(m[t, :, :]' * x[t, :], S[t, :, :]))
        m[t + 1, :, :] = rand(Distributions.MvNormal(m[t, :, :]), posterior_covariance(S[t, :, :], P[t, :, :]))
        P[t + 1, :, :] = P[t, :, :]
        S[t + 1, :, :] = S[t, :, :]
    end
    ssm.y = y
    return (y = y, m = m, P = P, S = S)
end

#simulate!(zeros(100,1), ones(100,1), p, h)