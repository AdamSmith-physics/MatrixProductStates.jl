
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

    A = copy(mps.tensors[site])  # vl, p, vr
    B = copy(mps.tensors[site+1])  # vl, p, vr
    C = contract(A, B, 3, 1)  # vl, p, p, vr
    C = reshape(C, (size(C,1), size(C,2)*size(C,3), size(C,4)))  # vl, p*p, vr
    C = contract(O, C, 2, 2)  # p, vl, vr
    C = permutedims(C, (2,1,3))
    C = reshape(C, (size(C,1)*mps.dim, mps.dim*size(C,3)))

    A, S, B = svd_truncated(C, mps.chiMax, mps.threshold, normalised=normalised)
    A = A * diag(S)
    mps.tensors[site] = reshape(A, (size(C,1),mps.dim,size(A,2)))
    mps.tensors[site+1] = reshape(B, (size(B,1),mps.dim,size(C,3)))
end