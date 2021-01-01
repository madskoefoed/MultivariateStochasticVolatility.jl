
# Predict output
output_mean(F, G, m) = m' * G' * F
output_covariance(Q, S, β, k) = Q * (1 - β) / (3β*k - 2k) * S

function output_predict(F, G, Q, m, S, β, k)
    μ = m' * G' * F
    Σ = Q * (1 - β) / (3β*k - 2k) * S
    return (μ, Σ)
end

# Predict state
function state_predict(F, G, P, Δ)
    R = Δ * G * P * G' * Δ
    Q = F' * R * F + 1.0
    K = R * F / Q
    return (R, Q, K)
end

# Update state
function state_update(G, K, Q, R, m, S, e, k)
    m = G * m + K * e'
    P = R - K * K' * Q
    S = S/k + e*e'/Q
    return (m, P, S)
end

prior_covariance(S, P, Δ) = kron(S, (Δ * P * Δ))
posterior_covariance(S, P) = kron(S, P)

standardized_error(y, μ, Σ) = inv(cholesky(Σ))*(y - μ)