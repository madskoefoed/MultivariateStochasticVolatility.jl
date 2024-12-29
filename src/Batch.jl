function update(model::MvStochVol, y::AbstractMatrix)
    models = MvStochVol[]
    for t in axes(y, 1)
        update!(model, y[t, :])
    end

    return nothing
end