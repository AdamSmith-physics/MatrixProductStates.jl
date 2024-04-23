using BenchmarkTools
using TensorOperations

export overlap

function overlap(A::MPS, B::MPS)
    if length(A) != length(B)
        throw(ArgumentError("MPS must have the same length."))
    end
    if size(A.tensors[1],1) != 1 || size(B.tensors[1],1) != 1 || size(A.tensors[end],3) != 1 || size(B.tensors[end],3) != 1
        throw(ArgumentError("Dangling indices must have dimension 1."))
    end


    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)  # set to 1 thread to avoid multi-threading overhead

    # compute overlap
    overlap = ones(1,1)
    for ii in 1:length(A)

        @tensor begin
            temp[la,lb,ra,rb] := conj(A.tensors[ii])[la,p,ra] * B.tensors[ii][lb,p,rb]
        end

        temp = reshape(temp, (size(temp,1)*size(temp,2), size(temp,3)*size(temp,4)))
        overlap = overlap * temp
    end

    BLAS.set_num_threads(blas_threads)  # reset to original number of threads

    return overlap[1,1]
end