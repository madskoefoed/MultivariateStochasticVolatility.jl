mutable struct Measurements
    y::Vector{<:Real}
    errors::Vector{Float64}
    scaled::Vector{Float64}
end

Measurements(p::Integer) = Measurements(zeros(p), zeros(p), zeros(p))