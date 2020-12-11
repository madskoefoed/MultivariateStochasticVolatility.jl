module MultivariateStochasticVolatility

# Import
import LinearAlgebra#: diag, kron, I
import Distributions#: Normal, MvNormal
#using StatsBase

# Aliases
#const A1R = Vector{<:Real}
#const A2R = Matrix{<:Real}
#const A3R = Array{<:Real, 3}

# Include scripts
include("./src/types.jl")
include("./src/utils.jl")
include("./src/estimate.jl")
include("./src/simulate.jl")

# Exported functions
export estimate, simulate!

end
