using Strided
using BenchmarkTools

export overlap

function overlap(A::MPS, B::MPS)
    if length(A) != length(B)
        throw(ArgumentError("MPS must have the same length."))
    end
    if size(A.tensors[1],1) != 1 || size(B.tensors[1],1) != 1 || size(A.tensors[end],3) != 1 || size(B.tensors[end],3) != 1
        throw(ArgumentError("Dangling indices must have dimension 1."))
    end

    # compute overlap
    overlap = ones(1,1)

    for ii in 1:length(A)
        temp = contract(conj(A.tensors[ii]), B.tensors[ii], 2, 2)
        temp = @strided permutedims(temp, (1,3,2,4))

        temp = reshape(temp, (size(temp,1)*size(temp,2), size(temp,3)*size(temp,4)))
        overlap = overlap * temp
        #temp = nothing
    end

    return overlap[1,1]
end