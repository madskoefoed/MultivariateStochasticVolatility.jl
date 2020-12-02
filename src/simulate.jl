
"""
simulate!(model::MultivariateModel)

Simulate a multivariate stochastic volatility model given the priors in ``model` with ``model.y`` updated in place.
"""

function simulate(m::AbstractVector{<:Real}, P::Real, S::AbstractMatrix{<:Real}; T::Integer = 1_000)
    p = length(m)
    y = zeros(T, p)
    for t in 1:T
        y[t, :] = rand(MvNormal(m, S))
        m = rand(MvNormal(m, S .* P))
    end
    return y
end
simulate(m::Real, P::Real, S::Real; T::Integer) = simulate([m], P, Matrix([S]'); T = T)