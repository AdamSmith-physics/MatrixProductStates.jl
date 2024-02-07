using Revise
using BenchmarkTools
using TensorOperations
using Strided
#using LinearAlgebra

# Provides utilities to contract and manipulate tensors

"""
    contract(x::Array{}, y::Array{}, idx1::Int, idx2::Int)
    contract(x::Array{}, y::Array{}, idxs1::Vector{Int}, idxs2::Vector{Int})

Contract two tensors across specified indexs.
"""
@inline function contract(x::Array{}, y::Array{}, idx1::Int, idx2::Int)
    sz = length(size(x))
    dims1 = [i == idx1 ? 0 : i for i = 1:sz]
    dims2 = [j == idx2 ? 0 : sz + j for j = 1:length(size(y))]
    return tensorcontract(x, dims1, y, dims2)
end

@inline function contract(x::Array{}, y::Array{}, idxs1::Vector{Int}, idxs2::Vector{Int})
    length(idxs1) != length(idxs2) && error("The length of contracting indexs differ.")
    labels = [-i for i = 1:length(idxs1)]
    dims1 = [i in idxs1 ? -findall(x -> x == i, idxs1)[1] : i for i = 1:length(size(x))]
    dims2 = [i in idxs2 ? -findall(x -> x == i, idxs2)[1] : i+length(size(x)) for i = 1:length(size(y))]
    return tensorcontract(x, dims1, y, dims2)
end


M1 = randn(1000,2000);
M2 = randn(2000,1000);

T1 = randn(100,100,2000);
T2 = randn(2000,100,100);

@benchmark M1 * M2  # 11.8ms on iMac
@benchmark contract(T1, T2, 3, 1)  # 1.466s on iMac


@benchmark T3 = permutedims(T1, (1,3,2))  # 37ms on iMac


A1 = randn(100,2,100);
A2 = randn(100,2,100);

@benchmark contract(A1, A2, 2, 2)  # 380ms on iMac
@benchmark out = permutedims(A1, (1,3,2))  # 72ms on iMac
@benchmark out = @strided permutedims(A1, (1,3,2))  # 48ns on iMac???

function perm(A::Array{Float64,3}, dims::Tuple{Int,Int,Int})
    out = @strided permutedims(A, dims)
    return out
end

@benchmark out = perm(A1, (1,3,2))  # 48ns on iMac???