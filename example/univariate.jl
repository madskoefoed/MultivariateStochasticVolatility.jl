"""
Example: a univariate time series with random noise of 1 in the state vector and output variance which increases by 1 every 100th observation from 1 to 10.
"""

# Packages
using Plots
using Distributions

# Generate a time series with changes in volatility every 100 observations
T = 1_000
Σ = vec([i for _ in 1:convert(Int, T/10), i = 1:10])
Φ = 10
x = zeros(T)
y = zeros(T, 1)
x[1] = rand(Normal(0, Φ))
y[1] = x[1] + rand(Normal(0, Σ[1]))
for t = 2:T
    x[t] = x[t - 1] + rand(Normal(0, Φ))
    y[t, 1] = x[t] + rand(Normal(0, Σ[t]))
end

# Struct containing y and priors
s = LocalLevel(y, ones(1, 1)*5, 10*ones(1, 1), ones(1, 1), 0.98, 0.98)

# Estimate
m = estimate(s)

# Plot simulated data and estimated means
scatter(s.y, color = :blue, markeralpha = 0.5, label = "Observations", legend = :topleft, markersize = 2)
hline!([0], color = :grey, linewidth = 2, label = "True mean")
plot!(m.μ, color = :red, linewidth = 2; label = "Estimated mean")

# Plot true and estimated variances
plot(m.Σ[:, 1, 1], color = :blue, linewidth = 1; label = "Estimated variance", legend = :topleft)
plot!(Σ .+ 1.0, color = :grey, linewidth = 2, label = "True variance")

# Plot estimated variances vs. variance of state
plot(m.Σ[:, 1, 1], color = :blue, linewidth = 1; label = "Measurement variance")
plot!(m.Φ[:, 1, 1], color = :grey, label = "State variance")

"""
Example: a univariate time series with local linear mean and trend and constant variance of 4.
"""