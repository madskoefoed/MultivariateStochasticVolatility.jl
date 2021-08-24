abstract type SSM end

mutable struct StateSpaceModel <: SSM
    y::REALMAT # T x p (response matrix)
    F::REALVEC # d x 1 (design vector)
    G::REALMAT # d x d (evolution matrix)

    m::REALMAT # d x p (parameter means)
    P::REALMAT # d x d (parameter covariance)
    S::REALMAT # p x p ()

    β::Real # Discount factor for covariance matrix
    δ::Real # Discount factor for parameters
    Δ::REALMAT
    ν::Real # Degrees of freedom of student-t distribution (calculated from β)
    n::Real #
    k::Real # Ensures that Σ is a random walk

    function StateSpaceModel(y, F, G, m, P, S, β, δ)
        p = size(y, 2)
        d = size(m, 1)

        # Check dimensions of the output equation
        !(p == size(m, 2) == size(S, 1) == size(S, 2)) && throw(DimensionMismatch("y has $p columns, m has $(size(m, 2)) columns, and S is a symmetric $(size(S)) matrix."))
        
        # Check dimensions of the state equation
        !(d == length(F) == size(P, 1) == size(P, 2) == size(G, 1) == size(G, 2)) && throw(DimensionMismatch("F has length $(length(F)), m has $(size(m, 1)) rows and P is a symmetric $(size(P)) matrix."))

        (δ  > 0 && δ <= 1) || throw(ArgumentError("0 < δ ≤ 1 required (currently $δ)."))
        (β > 2/3 && β < 1) || throw(ArgumentError("$(2//3) < β < 1 required (currently $β)."))
        n = 1/(1 - β)
        ν = β*n
        k = (β - p*β + p)/(2β - p*β + p - 1)
        Δ = diagm(fill(1/δ, d))
        return new(y, F, G, m, P, S, β, δ, Δ, ν, n, k)
    end
end

# Constructors for local level
function LocalLevel(y::REALMAT, m::REALMAT, P::REALMAT, S::REALMAT, β::Real, δ::Real)
    F = [1]
    G = ones(1, 1)
    return StateSpaceModel(y, F, G, m, P, S, β, δ)
end

# Constructors for local level trend
function LocalLevelTrend(y::REALMAT, m::REALMAT, P::REALMAT, S::REALMAT, β::Real, δ::Real)
    F = [1, 0]
    G = [1 1;
         0 1]
    return StateSpaceModel(y, F, G, m, P, S, β, δ)
end

# Constructors for local level cycle
function LocalLevelCycle(y::REALMAT, m::REALMAT, P::REALMAT, S::REALMAT, β::Real, δ::Real)
    F = [1, 1, 0]
    G = [1 0 0;
         0 0 0;
         0 0 0]
    return StateSpaceModel(y, F, G, m, P, S, β, δ)
end