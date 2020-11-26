module MultivariateStochasticVolatility

# Import
using LinearAlgebra: diag, kron
using Distributions: Normal, MvNormal
using StatsBase

# Include scripts
include("./src/util.jl")
include("./src/types.jl")
include("./src/estimate.jl")
include("./src/simulate.jl")

# Exported types and structs
export SVModel, UnivariateModel, MultivariateModel

# Exported functions
export estimate, simulate!

end
