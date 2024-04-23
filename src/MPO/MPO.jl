export MPO

mutable struct MPO <: AbstractMPO
    d::Int  # physical dimension
    N::Int  # number of sites

    tensors::Vector{Array{ComplexF64,4}}  # tensors (vl, p_out, p_in, vr)
end

"""
Default constructor for an empty MPO.

"""
MPO() = MPO(0, 0, []) 

"""
Default constructor for an N-site MPO with physical dimension d.
"""
MPO(N::Int, d::Int) = MPO(d, N, fill(zeros(ComplexF64, 1, d, d, 1), N))


export to_matrix
function to_matrix(O::MPO)::Matrix{ComplexF64}
    d = O.d
    N = O.N
    M = O.tensors[1]
    for i in 2:N
        M = contract(M, O.tensors[i], 4, 1)  # vl, p_outl, p_inl, p_outr, p_inr, vr
        M = permutedims(M, (1, 2, 4, 3, 5, 6))  # vl, p_outr, p_outl, p_inr, p_inl, vr
        M = reshape(M, size(M, 1), size(M, 2)*size(M, 3), size(M, 4)*size(M,5), size(M,6))
    end
    return reshape(M, d^N, d^N)
end