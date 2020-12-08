module MultivariateStochasticVolatility

# Import
using LinearAlgebra: diag, kron
using Distributions: Normal, MvNormal
#using StatsBase

# Include scripts
include("./src/types.jl")
include("./src/utils.jl")
include("./src/estimate.jl")
include("./src/simulate.jl")

# Exported functions
export estimate, simulate!

end
