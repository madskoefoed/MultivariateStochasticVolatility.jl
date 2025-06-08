abstract type AbstractPerformance end

mutable struct FilterPerformance <: AbstractPerformance
    LL::Float64
    ME::Vector{Float64}
    MAE::Vector{Float64}
    MSE::Vector{Float64}
    MSSE::Vector{Float64}
end

FilterPerformance(p::Integer) = FilterPerformance(0.0, zeros(p), zeros(p), zeros(p), zeros(p))