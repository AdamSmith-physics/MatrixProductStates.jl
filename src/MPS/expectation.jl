
export expectation_1site
function expectation_1site(mps::MPS, O::Array{<:Number,2}, site::Int)
    if site < 1 || site > length(mps)
        throw(ArgumentError("Invalid site for expectation value."))
    end
    movecentre!(mps, site)
    C = mps.tensors[site]  # vl, p, vr
    out = contract(O, C, 2, 2)  # p, vl, vr
    #out = contract(conj(C), out, 2, 1)  # vl*, vr*, vl, vr
    out = contract(conj(C), out, [1,2,3], [2,1,3])

    return out[1]  # return as number instead of zero dimensional array
end


export expectation_2site
function expectation_2site(mps::MPS, O::Array{<:Number,2}, site::Int)
    """
    Calculate the expectation value of a 2-site operator O acting on sites site and site+1 of the MPS.
    """
    if site < 1 || site > length(mps)-1
        throw(ArgumentError("Invalid site for expectation value."))
    end
    
    if site > mps.centre
        movecentre!(mps, site)
    elseif site+1 < mps.centre
        movecentre!(mps, site+1)
    end

    A = mps.tensors[site]  # vl, p, vr
    B = mps.tensors[site+1]  # vl, p, vr
    C = contract(A, B, 3, 1)  # vl, p, p, vr
    C = reshape(C, (size(C,1), size(C,2)*size(C,3), size(C,4)))  # vl, p*p, vr
    out = contract(O, C, 2, 2)  # p, vl, vr
    #out = contract(conj(C), out, 2, 1)  # vl*, vr*, vl, vr
    out = contract(conj(C), out, [1,2,3], [2,1,3])

    return out[1]  # return as number instead of zero dimensional array
end