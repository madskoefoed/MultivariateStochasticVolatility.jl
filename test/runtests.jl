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
@testset "Priors" begin
    p = 2
    m = [-1, 5]
    P = 999.9
    S = [10 0;
          0 1]

    priors = Priors(m, P, S)
    @test length(priors.m) == p
    @test length(priors.P) == 1
    @test size(priors.S, 1) == size(priors.S, 2)
    @test priors.S[1, 1] == 10
    @test priors.m[1] == -1
    @test sum(priors.m) == 4
    @test isapprox(priors.P, P)
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
    model  = MvStochVolFilter(priors, hp)
    
    estimate!(model, y)

    @test model.obs == T

    ### BATCH ###
    hp  = Hyperparameters()
    priors = Priors(m, P, S)
    model  = MvStochVolFilter(priors, hp)
    
    batch = estimate_batch!(model, y)
end

