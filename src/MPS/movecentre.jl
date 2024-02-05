

function moveright!(mps::MPS)
    if center(mps) < length(mps) && center(mps) >= 1
        A, S, B = svd(mps.tensors[mps.center], 3)  # exact SVD

        mps.tensors[mps.center] = A
        B = Diagonal(S) * B
        mps.tensors[mps.center+1] = contract(B, mps.tensors[mps.center+1], 2, 1)
        mps.center += 1
    end
end

function moveleft!(mps::MPS)
    if center(mps) > 1 && center(mps) <= length(mps)
        A, S, B = svd(mps.tensors[mps.center], 1)  # exact SVD

        mps.tensors[mps.center] = B
        A = A * Diagonal(S)
        mps.tensors[mps.center-1] = contract(mps.tensors[mps.center-1], A, 3, 1)
        mps.center -= 1
    end
end

function movecentre(mps::MPS, site::Int)
    if site < 1 || site > length(mps)
        throw(ArgumentError("Invalid site for center."))
    end

    while center(mps) < site
        moveright!(mps)
    end
    while center(mps) > site
        moveleft!(mps)
    end
end