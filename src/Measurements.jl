abstract type AbstractMeasurements end

mutable struct UnivariateMeasurements <: AbstractMeasurements
    y::Real
    errors::Float64
    scaled::Float64
end

UnivariateMeasurements() = UnivariateMeasurements(0.0, 0.0, 0.0)

mutable struct MultivariateMeasurements <: AbstractMeasurements
    y::Vector{<:Real}
    errors::Vector{Float64}
    scaled::Vector{Float64}
end

MultivariateMeasurements(p::Integer) = MultivariateMeasurements(zeros(p), zeros(p), zeros(p))