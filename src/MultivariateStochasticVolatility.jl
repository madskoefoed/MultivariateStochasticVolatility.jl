module MultivariateStochasticVolatility

# Import
using LinearAlgebra#: diag
using Distributions#: Normal, MvNormal, MvTDist
using StatsBase
#using PDMats

#using Plots

# Export
export estimate

# Include scripts
include("./src/util.jl")
include("./src/types.jl")
include("./src/estimate.jl")
#include("./src/simulate.jl")

end
