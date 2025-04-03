module MultivariateStochasticVolatility

# Code adapted from Forecasting with time-varying vector autoregressive models (2008), K. Triantafyllopoulos

# Import
using LinearAlgebra: diag, kron, I, cholesky, Diagonal, isposdef
using Distributions: Normal, MvNormal, MvTDist, MatrixNormal, logpdf
using PDMats: PDMat

# Include scripts
include("Hyperparameters.jl")
include("Parameters.jl")
include("Measurements.jl")
include("Performance.jl")
include("Models.jl")
include("Estimation.jl")
include("Simulate.jl")
include("Utils.jl")

# Exported types
export MvStochVol, Hyperparameters, Parameters
export MvNormal, MatrixNormal, MvTDist

# Exported functions
export estimate, estimate!
export simulate

end