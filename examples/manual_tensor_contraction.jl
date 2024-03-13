using Revise
using BenchmarkTools
using TensorOperations
using LoopVectorization

# create a random rank 3 complex tensor
L = 100

A = rand(ComplexF64, L, L, L);
B = rand(ComplexF64, L, L, L);

C = zeros(ComplexF64, L, L, L, L);

function contract(A, B, C)
    @tensor C[i, j, k, l] = A[i, p, k] * B[j, p, l];
    return C
end

function contract_explicit(A,B,C)
    @tturbo for i in 1:L
        for j in 1:L
            for k in 1:L
                for l in 1:L
                    @inbounds C[i,j,k,l] = 0.0
                    for p in 1:L
                       @inbounds C[i,j,k,l] += A[i,p,k] * B[j,p,l]
                    end
                end
            end
        end
    end
end


function contract_block(A, B, C)
    for k in 1:L
        for l in 1:L
            @inbounds C[:,:,k,l] = A[:,:,k] * transpose(B[:,:,l])
        end
    end
end

function contract_block_alt(A, B, C)
    for k in 1:L
        for j in 1:L
            @inbounds C[:,j,k,:] = A[:,:,k] * B[j,:,:]
        end
    end
end


# contract the tensors
@benchmark contract($A, $B, $C)

@benchmark contract_explicit($A, $B, $C)

@benchmark contract_block($A, $B, $C)

@benchmark contract_block_alt($A, $B, $C)