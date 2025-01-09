mutable struct Hyperparameters
    β::Real
    δ::Real
    ν::Float64
    κ::Float64

    function Hyperparameters(β::AbstractFloat, δ::AbstractFloat)
        (δ > 0   && δ <= 1) || throw(ArgumentError("0 < δ ≤ 1 required (currently $δ)."))
        (β > 2/3 && β  < 1) || throw(ArgumentError("$(2//3) < β < 1 required (currently $β)."))
        n = 1/(1 - β)
        ν = n * β
        κ = (1 - β) / (3*β - 2)
        
        return new(β, δ, ν, κ)
    end
end

Hyperparameters() = Hyperparameters(0.99, 0.99)

get_k(hyper::Hyperparameters, p::Integer) = (hyper.β - p*hyper.β + p)/(hyper.β - p*hyper.β + p - 1)
get_df(hyper::Hyperparameters)            = hyper.ν