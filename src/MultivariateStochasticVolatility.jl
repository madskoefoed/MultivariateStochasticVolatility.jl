module MultivariateStochasticVolatility

# Code based on Forecasting with time-varying vector autoregressive models (2008), K. Triantafyllopoulos

# Import
using LinearAlgebra: diag, kron, I, cholesky, Diagonal, isposdef
using Distributions: Normal, MvNormal, MvTDist, logpdf
using PDMats: PDMat

# Include scripts
include("Hyperparameters.jl")
include("Parameters.jl")
include("Predictive.jl")
include("Models.jl")
include("Estimate.jl")
include("Simulate.jl")

# Exported types
export MvStochVol, Hyperparameters, Priors, Posteriors
export PriorPredictive, MvTDist

# Exported functions
export estimate, estimate!, simulate

end