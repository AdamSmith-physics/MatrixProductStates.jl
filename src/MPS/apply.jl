
export apply_1site!
function apply_1site!(mps::MPS, O::Array{<:Number,2}, site::Int)
    if site < 1 || site > length(mps)
        throw(ArgumentError("Invalid site"))
    end
    movecentre!(mps, site)

    @tensor mps.tensors[site][vl, p, vr] := O[p, pc] * mps.tensors[site][vl, pc, vr]
end


export apply_2site!
"""
Apply 2 site gate to mps. Normalised truncation by default!
Automatically flips left-right to use lexigraphical ordering.
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

    O_copy = reshape(copy(O), (2,2,2,2))  # por, pol, pir, pio  (flipped!)

    @tensor theta[vl, pl, pr, vr] := mps.tensors[site][vl, pl, c] * mps.tensors[site+1][c, pr, vr]
    @tensor theta[vl, pl, pr, vr] := O_copy[pr, pl, cr, cl] * theta[vl, cl, cr, vr]  # flip left and right for contraction!
    vl = size(theta,1)
    vr = size(theta,4)
    theta = reshape(theta, (vl*mps.d, mps.d*vr))

    A, S, B = svd_truncated(theta, mps.chiMax, mps.threshold; normalised=normalised)
    if mps.centre == site
        A = A * diagm(S)
    elseif mps.centre == site+1
        B = diagm(S) * B
    else
        throw(ArgumentError("Centre site not updated correctly."))
    end

    mps.tensors[site] = reshape(A, (vl,mps.d,size(A,2)))
    mps.tensors[site+1] = reshape(B, (size(B,1),mps.d,vr))
    
end