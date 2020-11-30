
"""
simulate!(model::MultivariateModel)

Simulate a multivariate stochastic volatility model given the priors in ``model` with ``model.y`` updated in place.
"""

function simulate(S::AbstractMatrix{W}; T::Integer = 1_000) where {W<:Real}
    p = size(S, 1)
    y = zeros(T, p)
    m = zeros(p)
    for t in 1:T
        y[t, :] = rand(MvNormal(m, S))
    end
    return y
end
simulate(S::W; T::Integer) where {W<:Real} = simulate(Matrix([S]'); T = T)

function simulate(m::AbstractVector{W}, P::W, S::AbstractMatrix{W}; T::Integer = 1_000) where {W<:Real}
    p = length(m)
    y = zeros(T, p)
    for t in 1:T
        y[t, :] = rand(MvNormal(m, S))
        m = rand(MvNormal(m, S .* P))
    end
    return y
end
simulate(m::W, P::W, S::W; T::Integer) where {W<:Real} = simulate([m], P, Matrix([S]'); T = T)