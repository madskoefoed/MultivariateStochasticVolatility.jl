
#function check_inputs(β::Real, δ::Real)
#    if β >= 1 || β <= 2/3
#        throw(ArgumentError("β must be in the range ]2/3, 1[, is currently $β"))
#    elseif δ > 1 || δ <= 0
#        throw(ArgumentError("δ must be in the range ]0, 1], is currently $δ"))
#    end
#end

function estimate(Model::UnivariateModel, β::Real, δ::Real)

    y = Model.y

    # Constants
    T = size(y, 1)
    n = get_n(β)
    k = get_k(β, 1)
    
    m = zeros(T + 1) ; m[1] = Model.m
    P = zeros(T + 1) ; P[1] = Model.P
    S = zeros(T + 1) ; S[1] = Model.S

    μ = zeros(T)
    Σ = zeros(T)

    for t = 1:T
        R = P[t] / δ
        Q = R + 1.0
        K = R / Q
        
        μ[t] = predictive_mean(m[t])
        Σ[t] = predictive_covariance(Q, S[t], β, k)
        e    = predictive_error(y[t], μ[t])

        m[t + 1] = update_m(m[t], K, e)
        P[t + 1] = update_P(R, K, Q)
        S[t + 1] = update_S(S[t], Q, e, k)
    end
    return (μ = μ, Σ = Σ, m = m, P = P, S = S)
end

function estimate(Model::MultivariateModel, β::Real, δ::Real)

    y = Model.y

    # Constants
    T, p = size(y)
    n = get_n(β)
    k = get_k(β, p)
    
    m = zeros(T + 1, p)    ; m[1, :]    = Model.m
    P = zeros(T + 1)       ; P[1]       = Model.P
    S = zeros(T + 1, p, p) ; S[1, :, :] = Model.S

    μ = zeros(T, p)
    Σ = zeros(T, p, p)

    for t = 1:T
        R = P[t] / δ
        Q = R + 1.0
        K = R / Q

        μ[t, :]    = predictive_mean(m[t, :])
        Σ[t, :, :] = predictive_covariance.(Q, S[t, :, :], β, k)
        e          = predictive_error.(y[t, :], μ[t, :])

        m[t + 1, :]    = update_m.(m[t, :], K, e)
        P[t + 1]       = update_P(R, K, Q)
        S[t + 1, :, :] = update_S.(S[t, :, :], Q, e, k)
    end
    return (μ = μ, Σ = Σ, m = m, P = P, S = S)
end

x = cumsum(randn(100));
m = UnivariateModel(x, 0.0, 1000.0, 1.0);
f = estimate(m, 0.9, 0.9);

plot(x, color = :blue)
plot!(f.μ, color = :blue, linestyle = :dash)
#plot!(m.μ, color = :blue, linestyle = :dash)

X = [x cumsum(randn(100))];
M = MultivariateModel(X, [0., 0.], 1000.0, [1.0 0.0; 0.0 1.0]);
F = estimate(M, 0.9, 0.9);

plot(X, color = [:blue :red])
plot!(F.μ, color = [:blue :red], linestyle = [:dash :dash])