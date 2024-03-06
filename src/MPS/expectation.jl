using LoopVectorization
using Strided
using TensorOperations

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

export expectation
function expectation(mps::MPS, O::MPO)
    """
    Calculate the expectation value of the MPO O acting on the MPS.
    """
    N = length(mps)
    N == length(O) || throw(ArgumentError("MPS and MPO must have the same length."))

    matrices::Vector{Matrix{ComplexF64}} = [ones(ComplexF64,1,1) for _ in 1:N]

    
    perm(x) = permutedims(x, (1,3,5,2,4,6))
    for i in 1:N
        temp = contract(O[i], mps[i], 3, 2)  # Ol, Op, Or, ml, mr
        temp = contract(conj(mps[i]), temp, 2, 2)  # ml*, mr*, Ol, Or, ml, mr
        temp = perm(temp)  # ml*, Ol, ml, mr*, Or, mr
        #tensorcopy!(temp, TensorOperations.add_indices((1,2,3,4,5,6),(1,3,5,2,4,6)), copy(temp))
        #temp = tensorcopy((1,3,5,2,4,6), temp, (1,2,3,4,5,6))
        s_temp = size(temp)
        matrices[i] = reshape(temp, (s_temp[1]*s_temp[2]*s_temp[3], s_temp[4]*s_temp[5]*s_temp[6]))
    end

    """for i in 1:N
        temp = zeros(ComplexF64, size(mps[i],1), size(O[i],1), size(mps[i],1), size(mps[i],3), size(O[i],4), size(mps[i],3))
        @tensor begin
            temp[ml_c, Ol, ml, mr_c, Or, mr] = conj(mps[i])[ml_c, p1, mr_c] * O[i][Ol, p1, p2, Or] * mps[i][ml, p2, mr]
        end
        s_temp = size(temp)
        matrices[i] = reshape(temp, (s_temp[1]*s_temp[2]*s_temp[3], s_temp[4]*s_temp[5]*s_temp[6]))
    end"""

    """out = 1.0
    for i in 1:N
        out = out * matrices[i]
    end"""
    out = prod(matrices)


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
    id = [1 0; 0 1]

    H_local = zeros(ComplexF64, 4, 4)

    H_local += J * kron(X,X)
    H_local += h/2 * kron(Z, id) + h/2 * kron(id, Z)  # need to add extra at the ends

    E = 0.0
    for i in 1:N-1
        E += expectation_2site(mps, H_local, i)
    end

    E += h/2 * expectation_1site(mps, Z, 1)  # extra at the ends
    E += h/2 * expectation_1site(mps, Z, N)

    return E
end

export energy_TFIM2
function energy_TFIM2(mps::MPS, J::Real, g::Real, h::Real)
    """
    Calculate the energy of the transverse field Ising model with nearest neighbour coupling J and transverse field h.
    H = J * sum_i (Z_i Z_{i+1}) + h * sum_i Z_i + g * sum_i X_i
    """
    N = length(mps)
    X = [0 1; 1 0]
    Z = [1 0; 0 -1]
    id = [1 0; 0 1]

    H_local = zeros(ComplexF64, 4, 4)

    H_local += J * kron(Z,Z)
    H_local += h/2 * kron(Z, id) + h/2 * kron(id, Z)  # need to add extra at the ends
    H_local += g/2 * kron(X, id) + g/2 * kron(id, X)  # need to add extra at the ends

    E = 0.0
    for i in 1:N-1
        E += expectation_2site(mps, H_local, i)
    end

    E += h/2 * expectation_1site(mps, Z, 1)  # extra at the ends
    E += h/2 * expectation_1site(mps, Z, N)

    E += g/2 * expectation_1site(mps, X, 1)  # extra at the ends
    E += g/2 * expectation_1site(mps, X, N)

    return E

end