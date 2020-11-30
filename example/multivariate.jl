######################################
### Multivariate local level model ###
######################################

# Packages
using Plots

# Time series storage
y = zeros(500, 3)

# Struct containing y and priors
s = MultivariateModel(y, [0.0, -1.0, -2.0], 1.0, [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0])

# Simulate
simulate!(s)

# Estimate
m = estimate(s; β = 0.9, δ = 0.9)

# Plot simulated data and estimated means
scatter(s.y, color = [:blue :red :green], markeralpha = 0.5)
plot!(m.μ, color = [:blue :red :green], linewidth = 2; label = "")