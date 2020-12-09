module MultivariateStochasticVolatility

# Import
using LinearAlgebra#: diag, kron
using Distributions#: Normal, MvNormal
#using StatsBase

# Aliases
#const A1R = Vector{<:Real}
#const A2R = Matrix{<:Real}
#const A3R = Array{<:Real, 3}

# Include scripts
include("./src/types.jl")
include("./src/predict.jl")
include("./src/estimate.jl")
include("./src/simulate.jl")

# Exported functions
export estimate, simulate!

end
