using MultivariateStochasticVolatility

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

prior = MultivariateStochasticVolatility.Priors(m, P, S)

param = MultivariateStochasticVolatility.Parameters(prior, hyper);

# Model
model = MultivariateStochasticVolatility.MvStochVolFilter(param);

# Estimation
MultivariateStochasticVolatility.estimate!(model, y);

# Estimation with history
history = MultivariateStochasticVolatility.estimate_history!(model, y);


# Parameters
m = zeros(1);
P = 1000;
S = MultivariateStochasticVolatility.Diagonal(ones(1));

prior = MultivariateStochasticVolatility.Priors(m, P, S)

param = MultivariateStochasticVolatility.Parameters(prior, hyper);

# Model
model = MultivariateStochasticVolatility.MvStochVolFilter(param);

# Estimation
MultivariateStochasticVolatility.estimate!(model, y[:,1:1]);

# Estimation with history
history = MultivariateStochasticVolatility.estimate_history!(model, y);