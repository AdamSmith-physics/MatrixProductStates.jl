

function moveright!(mps::MPS; normalised::Bool=false)
    if centre(mps) < length(mps) && centre(mps) >= 1
        A, S, B = svd_tensor(mps.tensors[mps.centre], 3; normalised=normalised)  # exact SVD

        mps.tensors[mps.centre] = A
        B = Diagonal(S) * B
        mps.tensors[mps.centre+1] = contract(B, mps.tensors[mps.centre+1], 2, 1)
        mps.centre += 1
    end
end

function moveleft!(mps::MPS; normalised::Bool=false)
    if centre(mps) > 1 && centre(mps) <= length(mps)
        A, S, B = svd_tensor(mps.tensors[mps.centre], 1; normalised=normalised)  # exact SVD

        mps.tensors[mps.centre] = B
        A = A * Diagonal(S)
        mps.tensors[mps.centre-1] = contract(mps.tensors[mps.centre-1], A, 3, 1)
        mps.centre -= 1
    end
end

function movecentre!(mps::MPS, site::Int; normalised::Bool=false)
    if site < 1 || site > length(mps)
        throw(ArgumentError("Invalid site for center."))
    end

    while centre(mps) < site
        moveright!(mps; normalised=normalised)
    end
    while centre(mps) > site
        moveleft!(mps; normalised=normalised)
    end
end