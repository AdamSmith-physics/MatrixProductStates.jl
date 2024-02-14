
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

export energy_TFIM
function energy_TFIM(mps::MPS, J::Real, h::Real)
    """
    Calculate the energy of the transverse field Ising model with nearest neighbour coupling J and transverse field h.
    H = J * sum_i (X_i X_{i+1}) + h * sum_i Z_i
    """
    N = length(mps)
    X = [0 1; 1 0]
    Z = [1 0; 0 -1]
    I = [1 0; 0 1]

    H_local = zeros(ComplexF64, 4, 4)

    H_local += J * kron(X,X)
    H_local += h/2 * kron(Z, I) + h/2 * kron(I, Z)  # need to add extra at the ends

    E = 0.0
    for i in 1:N-1
        E += expectation_2site(mps, H_local, i)
    end

    E += h/2 * expectation_1site(mps, Z, 1)  # extra at the ends
    E += h/2 * expectation_1site(mps, Z, N)

    return E
end