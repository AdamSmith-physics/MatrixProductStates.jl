
export apply_1site!
function apply_1site!(mps::MPS, O::Array{<:Number,2}, site::Int)
    if site < 1 || site > length(mps)
        throw(ArgumentError("Invalid site"))
    end
    movecentre!(mps, site)

    C = mps.tensors[site]  # vl, p, vr
    C = contract(O, C, 2, 2)  # p, vl, vr
    mps.tensors[site] = permutedims(C, (2,1,3))  # vl, p, vr
end


export apply_2site!
"""
Apply 2 site gate to mps. Normalised truncation by default!
"""
function apply_2site!(mps::MPS, O::Array{<:Number,2}, site::Int; normalised::Bool=true)
    if site < 1 || site > length(mps)-1
        throw(ArgumentError("Invalid site"))
    end
    if site > mps.centre
        movecentre!(mps, site)
    elseif site+1 < mps.centre
        movecentre!(mps, site+1)
    end

    """O_swapped = reshape(O, (2,2,2,2))
    O_swapped = permutedims(O_swapped, (2,1,4,3))
    O_swapped = reshape(O_swapped, (4,4))

    A = copy(mps.tensors[site])  # vl, p, vr
    B = copy(mps.tensors[site+1])  # vl, p, vr
    C = contract(A, B, 3, 1)  # vl, p, p, vr
    C = reshape(C, (size(C,1), size(C,2)*size(C,3), size(C,4)))  # vl, p*p, vr
    C = contract(O_swapped, C, 2, 2)  # p, vl, vr
    C = permutedims(C, (2,1,3))
    vl = size(C,1)
    vr = size(C,3)
    C = reshape(C, (size(C,1)*mps.d, mps.d*size(C,3)))"""

    O_copy = reshape(copy(O), (2,2,2,2))
    A = mps.tensors[site]  # vl, p, vr
    B = mps.tensors[site+1]  # vl, p, vr
    A = contract(A, B, 3, 1)  # vl, pl, pr, vr
    A = contract(O_copy, A, [4,3], [2,3])  # pr, pl, vl, vr
    A = permutedims(A, (3,2,1,4))  # vl, pl, pr, vr
    vl = size(A,1)
    vr = size(A,4)
    A = reshape(A, (vl*mps.d, mps.d*vr))

    A, S, B = svd_truncated(A, mps.chiMax, mps.threshold; normalised=normalised)
    A = A * diagm(S)

    mps.tensors[site] = reshape(A, (vl,mps.d,size(A,2)))
    mps.tensors[site+1] = reshape(B, (size(B,1),mps.d,vr))
end