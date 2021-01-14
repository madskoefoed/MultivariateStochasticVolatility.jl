
"""
simulate(StateSpaceModel)

Simulate a stochastic volatility model. Volatility is NOT currently implemented as a random walk, but rather given by Σ.
"""

function simulate!(ssm::StateSpaceModel)
    y, F, G, m, P, S, β, δ, Δ, ν, n, k = ssm.y, ssm.F, ssm.G, ssm.m, ssm.P, ssm.S, ssm.β, ssm.δ, ssm.Δ, ssm.ν, ssm.n, ssm.k

    T, p = size(y)
    d = size(m, 1)

    # Priors (t = 0)
    Σ = rand(InverseWishart(n + 2p, S))
    Φ = rand(MatrixNormal(m, P, Σ))

    for t = 1:T
        # Sigma
        Σ = rand(InverseWishart(n + 2p, S))

        # State equation
        Φ = G * Φ + rand(MatrixNormal(zeros(d, p), P, Σ))

        # Measurement (observation) equation
        y[t, :] = F' * Φ + Σ
    end
end