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
    P = 999.9
    S = [10 0;
          0 1]
    param = Parameters(m, P, S)
    @test length(param.m) == p
    @test length(param.P) == 1
    @test size(param.S, 1) == size(param.S, 2)
    @test param.S[1,1] == 10
    @test param.m[1] == -1
    @test sum(param.m) == 4
    @test param.P > 0
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
    T = 50
    p = 3
    Φ = collect(p:-1:1)
    Σ = Diagonal(collect(1:p))
    y = simulate(T, Φ, Σ)

    hyper = Hyperparameters()
    param = Parameters(zeros(p), 1000.0, Matrix(Diagonal([1, 1, 1])))
    model = MvStochVol(param, hyper)
    
    estimate!(model, y)
end

