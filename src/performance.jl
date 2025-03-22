function goodness_of_fit(model::MvStochVol)

    denom = collect(size(model.e, 1))

    ME = cumsum(model.e; dims = 1) ./ denom
    MAE = cumsum(abs.(model.e); dims = 1) ./ denom
    RMSE = sqrt.(cumsum(model.e.^2; dims = 1) ./ denom)
    MSSE = cumsum(model.u.^2; dims = 1) ./ denom

    return (ME = ME, MAE = MAE, RMSE = RMSE, MSSE = MSSE)

end