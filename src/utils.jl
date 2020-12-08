
function update_state(P, δ)
    R = P/δ
    Q = R + 1.0
    K = R/Q
    return R, Q, K
end