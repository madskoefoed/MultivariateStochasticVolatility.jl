######################################
### Multivariate local level model ###
######################################

# Packages
using Plots
import Distributions

# Generate time series with a doubling in volatility after 500 observations
Σ = cholesky([1 -1; -1 2])
T = 500
y = randn(T, 2) * Σ.U
Σ = cholesky([2 -2; -2 4])
T = 500
y = vcat(y, randn(T, 2) * Σ.U)

# Struct containing y and priors
s = LocalLevel(y, [5 -5], 10*ones(1, 1), Matrix(LinearAlgebra.I, 2, 2), 0.95, 0.95)

# Estimate
m = estimate(s)

# Plot simulated data and estimated means
scatter(s.y, color = [:blue :red], markeralpha = 0.5, layout = (2, 1), label = "")
plot!(m.μ, color = [:blue :red], linewidth = 2; label = "")

# Plot true and estimated variances
plot([m.Σ[:, 1, 1] m.Σ[:, 2, 2]], color = [:blue :red], linewidth = 1; label = "", layout = (2, 1))
plot!([[ones(T);ones(T)*2] [ones(T)*2;ones(T)*4]], color = :grey, linewidth = 2)

# Plot estimated variances vs. variance of state
plot([m.Σ[:, 1, 1] m.Σ[:, 2, 2]], color = [:blue :red], linewidth = 1; label = "", layout = (2, 1))
plot!([m.Φ[:, 1, 1] m.Φ[:, 2, 2]], color = :grey, label = "")