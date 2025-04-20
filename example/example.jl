T = 100_000;
μ = [1, -2, 3, -4];
Σ = MultivariateStochasticVolatility.Diagonal([4, 3, 2, 1]);
y = MultivariateStochasticVolatility.simulate(T, μ, Σ);

# Hyperparameters
hyper = MultivariateStochasticVolatility.Hyperparameters(0.9999, 0.999);

# Parameters
m = zeros(4);
P = 1000;
S = MultivariateStochasticVolatility.Diagonal(ones(4));

param = MultivariateStochasticVolatility.MeanParameters(m, P, S, hyper);

# Model
model = MultivariateStochasticVolatility.Filter(param);

# Estimation
MultivariateStochasticVolatility.estimate!(model, y);

# Estimation with history
batch = MultivariateStochasticVolatility.batch!(model, y);

# Get parameters
MultivariateStochasticVolatility.get_parameters(model)

# Get predictions
MultivariateStochasticVolatility.get_predictions(model)