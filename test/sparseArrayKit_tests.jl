using Revise
using SparseArrayKit
using SparseArrays
using TensorOperations
using MatrixProductStates
using KrylovKit

X = [0 1; 1 0]

X_sp = SparseArray(X)

X_sp = reshape(X_sp, (2,2,1))

@tensor out[p1, p2, p3, p4] := X_sp[p1, p2, c] * X_sp[p3, p4, c]


CNOT = [1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
CNOT = reshape(CNOT, (2,2,2,2))
CNOT_sp = SparseArray(CNOT)



N = 20
J = 1
g = 1.4
h = 0.9

H_mpo = TFIM(N, -J, -g, -h)

H_full = to_matrix(H_mpo)
@time H_sp = to_sparse(H_mpo)

@time E_val_krylov, E_vec_krylov, _ = eigsolve(H_sp, 2^N, 1, :SR)