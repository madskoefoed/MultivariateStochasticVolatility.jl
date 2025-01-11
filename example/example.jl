#using MultivariateStochasticVolatility

T = 100_000
μ = [1, -2, 3, -4];
Σ = MultivariateStochasticVolatility.Diagonal([2, 2, 1, 1]);
y = MultivariateStochasticVolatility.simulate(T, μ, Σ);

# Hyperparameters
hyper = MultivariateStochasticVolatility.Hyperparameters(0.95, 0.95);

# Parameters
m = ones(4);
P = 10;
S = MultivariateStochasticVolatility.Diagonal(ones(4) * 1000);

# Model
model = MultivariateStochasticVolatility.MvStochVol(m, P, S, hyper);

# Estimation
MultivariateStochasticVolatility.estimate!(model, y);

# Batch model
#batch = MultivariateStochasticVolatility.estimate(model, y)