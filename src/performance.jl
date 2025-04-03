mutable struct Performance
    LL::Float64
    ME::Vector{Float64}
    MAE::Vector{Float64}
    MSE::Vector{Float64}
    MSSE::Vector{Float64}
end

Performance(p::Integer) = Performance(0.0, zeros(p), zeros(p), zeros(p), zeros(p))