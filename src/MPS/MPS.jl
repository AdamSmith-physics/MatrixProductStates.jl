
export MPS
export randomMPS

mutable struct MPS <: AbstractMPS
    d::Int  # physical dimension
    N::Int  # number of sites

    tensors::Vector{Array{ComplexF64,3}}  # tensors (vl, p, vr)
    centre::Int # centre site in [1, N]

    chiMax::Int  # maximum bond dimension (0 for unbounded)
    threshold::Float64  # truncation threshold (0.0 for no truncation)
end

"""
Default constructor for an empty MPS.

"""
MPS() = MPS(0, 0, [], 0, 0, 0.0) 

"""
Default constructor for an N-site MPS with physical dimension d, maximum bond dimension chiMax and truncation threshold threshold.
"""
function MPS(N::Int, d::Int, chiMax::Int, threshold::Float64)
    tensor = zeros(ComplexF64, 1, d, 1)
    tensor[1, 1, 1] = 1.0
    tensors = fill(tensor, N)
    MPS(d, N, tensors, 1, chiMax, threshold)
end

MPS(N::Int, d::Int) = MPS(N, d, 0, 0.0)  # default with no chiMax and threshold
MPS(N::Int) = MPS(N, 2, 0, 0.0)  # default with d=2, no chiMax and threshold

"""
Initialise an unnormalised MPS with random tensors of dimension d, N sites, maximum bond dimension chiMax and truncation threshold threshold.
"""
function randomMPS(N::Int, d::Int, chi::Int, chiMax::Int, threshold::Float64)
    tensors = [randn(ComplexF64, chi, d, chi) for _ in 1:N]
    tensors[1] = randn(ComplexF64, 1, d, chi)
    tensors[end] = randn(ComplexF64, chi, d, 1)
    MPS(d, N, tensors, 1, chiMax, threshold)
end

