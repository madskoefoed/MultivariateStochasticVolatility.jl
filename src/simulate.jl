
"""
simulate!(ss::StateSpace)

Simulate a stochastic volatility model given m, P, and S. If P > 0, the mean vector is simulated
as a random walk while P = 0 implies a standard stochastic volatility model without time-varying mean(s).
"""

function simulate!(ssm::StateSpace)
    y = ssm.y
    F = ssm.F
    G = ssm.G
    n = ssm.n
    # Constants
    T, p = size(y)
    d    = size(F, 2)
    # Storage
    m = zeros(T + 1, d, p)
    P = zeros(T + 1, d, d)
    S = zeros(T + 1, p, p)
    Σ = zeros(T + 1, p, p)
    Θ = zeros(T + 1, d, p)
    # Priors
    m[1, :, :] = ssm.m
    P[1, :, :] = ssm.P
    S[1, :, :] = ssm.S
    Σ[1, :, :] = rand(Distributions.InverseWishart(n + 2p, S[1, :, :]))
    Θ[1, :, :] = rand(Distributions.MatrixNormal(m[1, :, :], Σ[1, :, :], P[1, :, :]))
    # Loop through time: t = 1,...,T
    for t = 1:T
        # Draw S
        S[t + 1, :, :] = S[t, :, :]/k * (1 - β)/(3β - 2)
        # Draw Σ
        Σ[t + 1, :, :] = rand(Distributions.InverseWishart(n + 2p, S[t + 1, :, :]))

        # Draw Θ (state equation)
        Θ[t + 1, :, :] = simulate_Θ(Θ, G, Ω[], Σ[t, :, :])
        # Draw y (output equation)
        y[t, :] = simulate_y(Θ[t + 1, :, :], F[t, :], Σ)
        #Θ[t + 1, :, :] = 
        #P[t + 1, :, :] = P[t, :, :]
        #S[t + 1, :, :] = S[t, :, :]
    end
    ssm.y = y
    return (y = y, m = m, P = P, S = S)
end

#simulate!(zeros(100,1), ones(100,1), p, h)