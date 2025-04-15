abstract type AbstractPerformance end

mutable struct UnivariatePerformance <: AbstractPerformance
    LL::Float64
    ME::Float64
    MAE::Float64
    MSE::Float64
    MSSE::Float64
end

UnivariatePerformance() = UnivariatePerformance(0.0, 0.0, 0.0, 0.0, 0.0)

mutable struct MultivariatePerformance <: AbstractPerformance
    LL::Float64
    ME::Vector{Float64}
    MAE::Vector{Float64}
    MSE::Vector{Float64}
    MSSE::Vector{Float64}
end

MultivariatePerformance(p::Integer) = MultivariatePerformance(0.0, zeros(p), zeros(p), zeros(p), zeros(p))