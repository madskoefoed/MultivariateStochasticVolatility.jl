mutable struct MvStochVol
    predictions::Vector{PredictStep}
    updates::Vector{UpdateStep}
    const hyperparameters::Hyperparameters
    const p::Integer
    const k::Float64

    function MvStochVol(param::Parameters,
                        hyper::Hyperparameters)
        
        predictions = PredictStep[]
        updates     = UpdateStep[]

        push!(predictions, PredictStep(param, hyper))

        p = get_p(param)
        k = get_k(hyper, p)

        new(predictions, updates, hyper, p, k)
    end
end