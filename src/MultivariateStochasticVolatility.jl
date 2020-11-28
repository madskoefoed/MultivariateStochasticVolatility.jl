module MultivariateStochasticVolatility

# Import
using LinearAlgebra: diag, kron
using Distributions: Normal, MvNormal
using StatsBase

# Include scripts
include("./util.jl")
include("./types.jl")
include("./estimate.jl")
include("./simulate.jl")

# Exported types and structs
export SVModel, UnivariateModel, MultivariateModel

# Exported functions
export estimate, simulate!

end
