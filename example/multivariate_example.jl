#using MultivariateStochasticVolatility

T = 100_000;
μ = [1, -2, 3, -4];
Σ = MultivariateStochasticVolatility.Diagonal([4, 3, 2, 1]);
y = MultivariateStochasticVolatility.simulate(T, μ, Σ);

# Hyperparameters
hp = MultivariateStochasticVolatility.Hyperparameters(0.9999, 0.999);

# Parameters
m = zeros(4);
P = 1000;
S = MultivariateStochasticVolatility.Diagonal(ones(4));

priors = MultivariateStochasticVolatility.Priors(m, P, S);

# Model
model = MultivariateStochasticVolatility.StochVolFilter(priors, hp);

# Estimation
MultivariateStochasticVolatility.estimate!(model, y);

# Estimation with history
batch = MultivariateStochasticVolatility.estimate_batch!(model, y);

# Get parameters
MultivariateStochasticVolatility.get_parameters(model)

# Get predictions
MultivariateStochasticVolatility.get_predictions(model)