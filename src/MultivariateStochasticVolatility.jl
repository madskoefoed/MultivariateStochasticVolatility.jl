module MultivariateStochasticVolatility

# Import
using LinearAlgebra: diag, diagm, kron, I, cholesky
using Distributions: Normal, MvNormal, InverseWishart, MatrixNormal

# Constants
const REALMAT = Matrix{T} where T <:Real
const REALVEC = Vector{T} where T <:Real

# Include scripts
include("types.jl")
include("utils.jl")
include("estimate.jl")
include("simulate.jl")

# Exported types
export StateSpace, LocalLevel, LocalLevelTrend

# Exported functions
export estimate

end
