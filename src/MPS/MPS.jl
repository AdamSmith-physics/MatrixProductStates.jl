
export MPS

mutable struct MPS <: AbstractMPS
    d::Int  # physical dimension
    N::Int  # number of sites

    tensors::Vector{Array{ComplexF64,3}}  # tensors (vl, p, vr)
    centre::Int # centre site in [1, N]

    chiMax::Int  # maximum bond dimension (0 for unbounded)
    threshold::Float64  # truncation threshold (0.0 for no truncation)
end

MPS() = MPS(0, 0, [], 0, 0, 0.0)


"""
Default constructor for an N-site MPS with physical dimension 2 (qubits / spin-1/2).
"""
function MPS(N::Int)
    tensor = zeros(ComplexF64, 1, 2, 1)
    tensor[1, 1, 1] = 1.0
    tensors = fill(tensor, N)
    MPS(2, N, tensors, 1, 0, 0.0)
end