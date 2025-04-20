using MultivariateStochasticVolatility
using Test
#using LinearAlgebra

#######################
### Hyperparameters ###
#######################
@testset "Hyperparameters" begin
    h = Hyperparameters(0.999, 0.2)
    @test isapprox(h.β, 0.999)
    @test isapprox(h.δ, 0.2)
    @test isapprox(h.ν, h.β/(1 - h.β))
    @test isapprox(h.κ, (1 - h.β) / (3*h.β - 2))

    h = Hyperparameters(0.8, 0.9)
    @test isapprox(h.β, 0.8)
    @test isapprox(h.δ, 0.9)
    @test isapprox(h.ν, h.β/(1 - h.β))
    @test isapprox(h.κ, (1 - h.β) / (3*h.β - 2))
end

##################
### Parameters ###
##################
@testset "Parameters" begin
    m = [-1, 5]
    P = 999.9
    S = [10 0;
          0 1]
    p = length(m)

    h = Hyperparameters(0.8, 0.9)
    param = Parameters(m, P, S, h)
    @test length(param.m) == p
    @test length(param.P) == 1
    @test size(param.S, 1) == size(param.S, 2)
    @test param.S[1, 1] == 10
    @test param.m[1] == -1
    @test sum(param.m) == 4
    @test isapprox(param.P, P)

    # Test weak priors
    param = Parameters(p, h)
    @test length(param.m) == p
    @test length(param.P) == 1
    @test size(param.S, 1) == size(param.S, 2)
    @test sum(param.m) == 0
    @test param.P == 1e6
    @test sum(param.S[i, i] for i in 1:p) == p
end

################
### Simulate ###
################
@testset "Simulate" begin
    T = 50
    p = 3
    Φ = collect(p:-1:1)
    Σ = [i == j ? 1.0 : 0.0 for i in 1:p, j in 1:p]
    y = simulate(T, Φ, Σ)

    @test size(y, 1) == T
    @test size(y, 2) == p
end

################
### Estimate ###
################
@testset "Online estimation" begin
    T = 50
    p = 3
    Φ = collect(p:-1:1)
    Σ = [i == j ? 1.0 : 0.0 for i in 1:p, j in 1:p]
    y = simulate(T, Φ, Σ)

    # Parameters
    m = zeros(p)
    P = 1000.0
    S = [i == j ? 1.0 : 0.0 for i in 1:p, j in 1:p]

    hyper = Hyperparameters(0.9, 0.9)
    param = Parameters(m, P, S, hyper)
    model = Filter(param)
    
    estimate!(model, y)

    @test model.obs == T
end

@testset "Batch estimation" begin
    T = 50
    p = 3
    Φ = collect(p:-1:1)
    Σ = [i == j ? 1.0 : 0.0 for i in 1:p, j in 1:p]
    y = simulate(T, Φ, Σ)

    # Parameters
    m = zeros(p)
    P = 1000.0
    S = [i == j ? 1.0 : 0.0 for i in 1:p, j in 1:p]

    hyper = Hyperparameters(0.99, 0.99)
    param = Parameters(m, P, S, hyper)
    model = Filter(param)
    
    batch = batch!(model, y)
end

