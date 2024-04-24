
export MPS
export randomMPS
export vector2MPS

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

"""
Initialise an MPS from a state vector psi.
"""
function vector2MPS(psi::Vector{ComplexF64}, d::Int, chiMax::Int, threshold::Float64; normalised::Bool=true)
    N = Int(log2(length(psi))/log2(d))

    tensors = []
    chi_old = 1
    phi = copy(psi)

    for _ in 1:N-1

        phi = reshape(phi, (chi_old*d, :))

        U, S, phi = svd_truncated(phi, chiMax, threshold; normalised=normalised)
        chi_new = length(S)
        phi = S .* phi

        append!(tensors, [reshape(U, chi_old, d, chi_new)])

        chi_old = chi_new
    end

    append!(tensors, [reshape(phi, (chi_old, d, 1))])

    MPS(d, N, tensors, N, chiMax, threshold)
end


export flatten
"""
Contract and flatten the MPS into a vector.
"""
function flatten(mps::MPS)

    # contract the MPS into a single tensor
    tensor = ones(ComplexF64, 1, 1)
    for i in 1:mps.N
        @tensor tensor[p1,p2,vr] := tensor[p1, c] * mps.tensors[i][c, p2, vr]
        tensor = reshape(tensor, size(tensor, 1)*size(tensor, 2), size(tensor, 3))  # (p1*p2), vr
    end

    tensor = reshape(tensor, size(tensor, 1))  # (p1*p2)*vr  # will fail if the end tensor doesn't have dangling index with dim 1

    return tensor
end

