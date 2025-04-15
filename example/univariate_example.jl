#using MultivariateStochasticVolatility

T = 100_000;
μ = -3;
Σ = 2;
y = MultivariateStochasticVolatility.simulate(T, μ, Σ);

# Hyperparameters
hp = MultivariateStochasticVolatility.Hyperparameters(0.9999, 0.999);

# Parameters
m = 0;
P = 1000;
S = 1;

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