
# Helper funtions
get_R(P, G, Δ) = sqrt.(Δ) * G * P * G' * sqrt.(Δ)
get_Q(F, R) = F' * R * F + 1.0
get_K(F, R, Q) = R * F / Q
get_m(m, G, K, e) = G * m + K * e'
get_P(R, K, Q) = R - K * K' * Q
get_S(S, k, e, Q) = S/k + e*e'/Q

function diagnostics(ssm::StateSpaceModel, output)
    μ = output.predicted.μ
    Σ = output.predicted.Σ
    e = output.predicted.e
    u = output.predicted.u
    ME = mean(e; dims = 1)
    MAE = mean(abs.(e); dims = 1)
    MSSE = mean(u.^2; dims = 1)
    LL = mean([logpdf(MvTDist(ssm.ν, μ[t, :], Σ[t, :, :]), y[t, :]) for t in 1:size(ssm.y, 1)])
    return (ME = ME, MAE = MAE, MSSE = MSSE, LogLik = LL)
end