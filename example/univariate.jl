####################################
### Univariate local level model ###
####################################

# Load packages
using Plots

# Time series storage
y = zeros(500)

# Struct containing y and priors
s = UnivariateModel(y, 0.0, 1.0, 1.0)

# Simulate
simulate!(s)

# Estimate
m = estimate(s; β = 0.9, δ = 0.9)

# Plot simulated data and estimated mean
scatter(s.y, color = :blue, markeralpha = 0.5)
plot!(m.μ, color = :blue, linewidth = 2, label = "")