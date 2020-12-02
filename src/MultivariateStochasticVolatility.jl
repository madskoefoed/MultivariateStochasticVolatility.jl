module MultivariateStochasticVolatility

# Import
using LinearAlgebra: diag, kron
using Distributions: Normal, MvNormal
using StatsBase

# Include scripts
include("./estimate.jl")
include("./simulate.jl")

# Exported functions
export estimate, simulate!

end
