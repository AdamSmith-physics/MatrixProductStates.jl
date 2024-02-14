"""
Change the chiMax and threshold of an MPS and truncate it by sweeping through the MPS from left to right and right to left.
"""
function truncate!(mps::MPS, chiMax::Int, threshold::Float64; normalised::Bool=false)
    # start at left end
    movecentre!(mps, 1)

    mps.chiMax = chiMax
    mps.threshold = threshold

    movecentre!(mps, length(mps); normalised=normalised)
end