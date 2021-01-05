"""
Example: a univariate time series with local linear mean and trend and constant variance of 4.
"""

T = 1_000
Σ = 0.5
Φ = [0.1 0.0; 0.0 0.1]
x = zeros(T, 2)
y = zeros(T, 1)
x[1, :] = rand(MvNormal([-1, 1], Φ))
y[1] = sum(x[1, :]) + rand(Normal(0, Σ[1]))
for t = 2:T
    x[t, :] = [-1, 1] + rand(MvNormal([0, 0], Φ))
    y[t, 1] = sum(x[t, :]) + rand(Normal(0, Σ))
end

# Struct containing y and priors
F = [1, 0]
G = Matrix(I, 2, 2)
m = zeros(2, 1)
P = Matrix(I, 2, 2) * 1000
S = Matrix(I, 1, 1)

s = StateSpaceModel(y, F, G, m, P, S, 0.95, 0.95)

# Estimate
m = estimate(s)

plot([y m.predicted.μ m.filtered.μ], labels = ["True" "Predicted" "Filtered"], legend = :topleft)