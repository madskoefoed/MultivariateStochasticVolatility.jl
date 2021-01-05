module MultivariateStochasticVolatility

# Import
using LinearAlgebra #: diag, kron, I
using Distributions #: Normal, MvNormal
using Plots

# Include scripts
include("./src/types.jl")
include("./src/utils.jl")
include("./src/estimate.jl")
#include("./src/simulate.jl")

# Exported types
export StateSpace, LocalLevel, LocalLevelTrend

# Exported functions
export estimate

end
