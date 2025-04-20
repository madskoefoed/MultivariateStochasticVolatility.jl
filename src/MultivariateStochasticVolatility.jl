module MultivariateStochasticVolatility

# Code adapted from Forecasting with time-varying vector autoregressive models (2008), K. Triantafyllopoulos

# Import
using LinearAlgebra: cholesky, Diagonal, isposdef
using Distributions: Normal, MvNormal, MvTDist, logpdf

# Include scripts
include("Hyperparameters.jl")
include("Parameters.jl")
include("Performance.jl")
include("Models.jl")
include("Estimation.jl")
include("Simulate.jl")
include("Utils.jl")

# Exported types
export Filter, Hyperparameters
export MeanParameters
export MatrixNormal, MvTDist

# Exported functions
export batch!, estimate!
export simulate

end