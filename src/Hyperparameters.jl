struct Hyperparameters
    β::Real
    δ::Real
    ν::Float64
    κ::Float64

    function Hyperparameters(β::AbstractFloat, δ::AbstractFloat)
        (δ > 0   && δ <= 1) || throw(ArgumentError("0 < δ ≤ 1 required (currently $δ)."))
        (β > 2/3 && β  < 1) || throw(ArgumentError("$(2//3) < β < 1 required (currently $β)."))
        ν = β/(1 - β)
        κ = (1 - β) / (3β - 2)
        
        return new(β, δ, ν, κ)
    end
end

get_k(h::Hyperparameters, p::Integer) = (h.β - p*h.β + p)/(2h.β - p*h.β + p - 1)
get_df(h::Hyperparameters)            = h.ν