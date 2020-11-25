"""

"""

function simulate(UnivariateModel, T = 1_000)
    x = zeros(T)
    m = zeros(T + 1)
    m[1] = UnivariateModel.m
    P = UnivariateModel.P
    S = UnivariateModel.S
    for t in 1:T
        x[t] = m[t] + rand(Normal(0, S))
        m[t + 1] = m[t] + rand(Normal(0, S * P))
    end
    return (x = x, m = m)
end

function simulate(MultivariateModel, T = 1_000)
    x = zeros(T, p)
    p = size(MultivariateModel.m)
    m = zeros(T + 1, p)
    m[1, :] = MultivariateModel.m
    P = MultivariateModel.P
    S = MultivariateModel.S
    for t in 1:T
        x[t, :] = m[t, :] + rand(MvNormal(0, S))
        m[t + 1, :] = m[t, :] + rand(MvNormal(0, kron(S, P)))
    end
    return (x = x, m = m)
end

u = UnivariateModel(0, 1, 1)
s = simulate(u, 50)
plot(s.x)
plot!(s.m)

U = MultivariateModel(0, 1, 1)
S = simulate(U, 50)
plot(s.x)
plot!(s.m)