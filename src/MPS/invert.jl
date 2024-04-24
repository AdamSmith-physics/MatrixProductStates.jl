

export invert!
"""
Spatial inversion of MPS
"""
function invert!(mps::MPS)
    tensors = reverse(mps.tensors)

    for i in eachindex(tensors)
        tensors[i] = permutedims(tensors[i], (3,2,1))
    end

    mps.tensors = tensors
    mps.centre = mps.N - mps.centre + 1
end

export invert
function invert(mps::MPS)
    mps_copy = deepcopy(mps)
    invert!(mps_copy)
    return mps_copy
end