function simulate(T::Integer, Φ::Real, Σ::Real)
    @assert T > 0 "The number of observations, T, must be strictly positive."
    @assert Σ > 0 "The variance, Σ, must be strictly positive."

    y = rand(Normal(Φ, sqrt(Σ)), T)
    
    return y
end

function simulate(T::Integer, Φ::Vector{<:Real}, Σ::AbstractMatrix)
    @assert T > 0 "The number of observations must be strictly positive."
    @assert isposdef(Σ) "The covariance matrix, Σ, must be positive definite."

    p = length(Φ)
    y = zeros(T, p)
    
    str = "Φ is a $(p)-dimensional vector,\nwhile Σ is a $(size(Σ, 1)) x $(size(Σ, 2)) matrix."
    !(p == size(Σ, 1) == size(Σ, 2)) && throw(DimensionMismatch(str))

    for t in axes(y, 1)
        y[t, :] = rand(MvNormal(Φ, Σ))
    end

    return y
end

# Quick simulation
function simulation()
    T = 1_000
    Φ = [-1, 2, -3]
    Σ = Diagonal([3, 2, 1])

    y = simulate(T, Φ, Σ)

    return y
end