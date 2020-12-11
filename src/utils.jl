prior_covariance(S, P, Δ) = LinearAlgebra.kron(S * (Δ * P * Δ))
posterior_covariance(S, P) = LinearAlgebra.kron(S * P)

#draw_m()
#draw_P()

num2mat(x::Number) = repeat([x], 1, 1)
vec2mat(x::AbstractVector) = repeat(x, 1, 1)