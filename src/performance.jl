mutable struct Performance
    loglikelihood::Float64
    mean_error::Vector{Float64}
    mean_absolute_error::Vector{Float64}
    mean_squared_error::Vector{Float64}
    mean_squared_standardized_error::Vector{Float64}
end

Performance(p::Integer) = Performance(0.0, zeros(p), zeros(p), zeros(p), zeros(p))