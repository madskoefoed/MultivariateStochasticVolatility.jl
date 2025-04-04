abstract type FilterModel end
struct StochVolModel{T <: AbstractParameters} <: FilterModel
    hyperparameters::Hyperparameters
    parameters::Vector{T}
    measurements::Vector{Measurements}
    p::Integer
    k::Float64
    #performance::Performance

    function FilterModel{T}(hyperparameters::Hyperparameters,
                            parameters::T) where {T <: AbstractParameters}

        p = size(parameters.S, 1)
        k = get_k(hyper, p)
        parameters = Vector{T}[parameters]

        new(hyperparameters, parameters, Measurements[], p, k)
    end
end