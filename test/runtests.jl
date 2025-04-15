using MultivariateStochasticVolatility
using Test
using LinearAlgebra

#######################
### Hyperparameters ###
#######################
@testset "Hyperparameters" begin
    h = Hyperparameters()
    @test isapprox(h.β, 0.99)
    @test isapprox(h.δ, 0.99)

    h = Hyperparameters(0.8, 0.9)
    @test isapprox(h.β, 0.8)
    @test isapprox(h.δ, 0.9)
end

##################
### Parameters ###
##################
@testset "Parameters" begin
    p = 2
    m = [-1, 5]
    R = 999.9
    P = 999.9
    S = [10 0;
          0 1]
    param = Priors(m, R, S)
    @test length(param.m) == p
    @test length(param.R) == 1
    @test size(param.S, 1) == size(param.S, 2)
    @test param.S[1, 1] == 10
    @test param.m[1] == -1
    @test sum(param.m) == 4
    @test isapprox(param.R, R)
    param = Posteriors(m, P, S)
    @test length(param.m) == p
    @test length(param.P) == 1
    @test size(param.S, 1) == size(param.S, 2)
    @test param.S[1, 1] == 10
    @test param.m[1] == -1
    @test sum(param.m) == 4
    @test isapprox(param.P, P)
end

################
### Simulate ###
################
@testset "Simulate" begin
    T = 50
    p = 3
    Φ = collect(p:-1:1)
    Σ = Diagonal(collect(1:p))
    y = simulate(T, Φ, Σ)

    @test size(y, 1) == T
    @test size(y, 2) == p
end

################
### Estimate ###
################
@testset "Estimate" begin
    ### ONLINE ###
    T = 50
    p = 3
    Φ = collect(p:-1:1)
    Σ = Diagonal(collect(1:p))
    y = simulate(T, Φ, Σ)
    m = zeros(p)
    P = 1000.0
    S = Diagonal(ones(p))

    hp  = Hyperparameters()
    priors = Priors(m, P, S)
    model  = MvStochVol(priors, hp)
    
    estimate!(model, y)

    @test model.observations == T

    ### BATCH ###
    hp  = Hyperparameters()
    priors = Priors(m, P, S)
    model  = MvStochVol(priors, hp)
    
    batch = estimate(model, y)
end

