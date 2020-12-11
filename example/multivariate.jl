######################################
### Multivariate local level model ###
######################################

# Packages
using Plots
import Distributions

# Time series storage
y = zeros(500, 3)

# Hyperparameters
h = Hyperparameters(0.9, 0.9)

# Priors
p = Priors(zeros(1, 3), repeat([100.], 1, 1), Matrix(LinearAlgebra.I, 3, 3))

# Struct containing y and priors
s = LocalLevel(y, p, h)

# Simulate
simulate!(s)

# Estimate
m = estimate(s; β = 0.9, δ = 0.9)

# Plot simulated data and estimated means
scatter(s.y, color = [:blue :red :green], markeralpha = 0.5)
plot!(m.μ, color = [:blue :red :green], linewidth = 2; label = "")