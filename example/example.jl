#using MultivariateStochasticVolatility

T = 1_000
μ = [1, -2, 3, -4];
Σ = MultivariateStochasticVolatility.Diagonal([2, 2, 1, 1]);
y = MultivariateStochasticVolatility.simulate(T, μ, Σ);

# Hyperparameters
h = MultivariateStochasticVolatility.Hyperparameters(0.95, 0.95);

# Parameters
m = [0, 0];
P = 10;
S = MultivariateStochasticVolatility.Diagonal(ones(2) * 1000);
param = MultivariateStochasticVolatility.Parameters(m, P, S);

# Model
model = MultivariateStochasticVolatility.MvStochVol(param, h);

# Estimation
MultivariateStochasticVolatility.estimate!(model, y);

# Batch model
batch = MultivariateStochasticVolatility.update(model, y)