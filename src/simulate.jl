"""

"""

function simulate!(model::UnivariateModel)
    T = length(model.y)
    m = model.m
    P = model.P
    S = model.S
    for t in 1:T
        model.y[t] = rand(Normal(m, S))
        m = rand(Normal(m, S * P))
    end
end

function simulate!(model::MultivariateModel)
    T, p = size(model.y)
    m = model.m
    P = model.P
    S = model.S
    for t in 1:T
        model.y[t, :] = rand(MvNormal(m, S))
        m = rand(MvNormal(m, S .* P))
    end
end

u = UnivariateModel(zeros(100), 0.0, 4.0, 1.0)
simulate!(u)
plot(u.y)

U = MultivariateModel(zeros(100, 3), [0.0, -1.0, -2.0], 3.0, [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 4.0])
simulate!(U)
plot(U.y)