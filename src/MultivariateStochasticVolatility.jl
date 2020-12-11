module MultivariateStochasticVolatility

# Import
import LinearAlgebra#: diag, kron, I
import Distributions#: Normal, MvNormal

# Include scripts
include("./src/types.jl")
include("./src/utils.jl")
include("./src/estimate.jl")
include("./src/simulate.jl")

# Exported functions
export estimate, simulate!

end
