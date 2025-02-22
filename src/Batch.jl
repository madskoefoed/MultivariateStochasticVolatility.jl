mutable struct MvStochVolBatch <: MvStochVol
    m::Matrix{Float64}
    P::Vector{Float64}
    S::Array{Float64, 3}
    μ::Matrix{Float64}
    Σ::Array{Float64, 3}

    y::Matrix{<:Real}
    errors::Matrix{Float64}
    scaled::Matrix{Float64}

    loglik::Vector{Float64}
    obs::Integer

    function MvStochVolBatch(parameters::Parameters,
                             y::Matrix{<:Real})

        T, p = size(y)

        @assert p == size(parameters.S, 1) "The dimensions of 'y' and 'S' do not match"

        # Create storage
        m = zeros(T+1, p)
        P = zeros(T+1)
        S = zeros(T+1, p, p)
        μ = zeros(T+1, p)
        Σ = zeros(T+1, p, p)

        errors = zeros(T+1, p)
        scaled = zeros(T+1, p)

        loglik = zeros(T)

        m[1, :]    = parameters.m
        P[1]       = parameters.P
        S[1, :, :] = parameters.S

        μ[1, :]    = parameters.m
        Σ[1, :, :] = (parameters.P + 1) * (1 - parameters.hyper.β) / (3*parameters.hyper.β - 2) * parameters.S

        for t in axes(y, 1)

            # Update t|t
            errors[t, :] = y[t, :] - μ[t, :]
            scaled[t, :] = inv(cholesky(Σ[t, :, :])) * errors[t, :, :]
            loglik[t]    = logpdf(MvTDist(parameters.hyper.ν, μ[t, :], Σ[t, :, :]), y[t, :])
        
            # Update at time t|t
            Q = P[t] + 1
            K = P[t] / Q

            m[t+1, :]    = m[t, :] + K * errors[t, :]
            P[t+1]       = (P[t] - (K * K') * Q) / parameters.hyper.δ
            S[t+1, :, :] = (S[t, :, :] + (errors[t, :]*errors[t, :]')/Q) / parameters.k

            μ[t+1, :]    = m[t+1, :]
            Σ[t+1, :, :] = (P[t+1] + 1) * (1 - parameters.hyper.β) / (3*parameters.hyper.β - 2) * S[t+1, :, :]

        end

        loglik = cumsum(loglik)

        new(m, P, S, μ, Σ, y, errors, scaled, loglik, T)
    end
end