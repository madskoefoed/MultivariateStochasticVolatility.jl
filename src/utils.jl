prior_covariance(S, P, Δ) = LinearAlgebra.kron(S, (Δ * P * Δ))
posterior_covariance(S, P) = LinearAlgebra.kron(S, P)

function simulate_Θ(Θ, G, Ω, Σ)
    Θ = G * Θ + rand(MatrixNormal(Σ, Ω))
    return Θ
end

function simulate_y(Θ, F, Σ)
    y = F' * Θ + rand(MvNormal(Σ))
    return y
end

num2mat(x::Number) = fill(x, 1, 1)
vec2mat(x::AbstractVector) = repeat(x, 1, 1)