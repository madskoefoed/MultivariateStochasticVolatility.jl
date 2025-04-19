module MultivariateStochasticVolatility

# Code adapted from Forecasting with time-varying vector autoregressive models (2008), K. Triantafyllopoulos

# Import
using LinearAlgebra: diag, kron, I, cholesky, Diagonal, isposdef, eigen
using Distributions: Normal, TDist, MvNormal, MvTDist, MatrixNormal, logpdf
using PDMats: PDMat

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
export MvNormal, MatrixNormal, MvTDist, TDist

# Exported functions
export estimate_batch!, estimate!
export simulate

end