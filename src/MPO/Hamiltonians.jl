export TFIM
function TFIM(N::Int, J::Real, h::Real)::MPO

    Id = [1.0 0.0; 0.0 1.0]
    X = [0.0 1.0; 1.0 0.0]
    Z = [1.0 0.0; 0.0 -1.0]

    O = MPO(N, 2)
    W = zeros(ComplexF64, 3, 2, 2, 3)
    W[1, :, :, 1] = Id
    W[1, :, :, 2] = X
    W[1, :, :, 3] = h*Z
    W[2, :, :, 3] = J*X
    W[3, :, :, 3] = Id

    for i in 1:N
        O.tensors[i] = copy(W)
    end
    O[1] = reshape(O[1][1, :, :, :], 1, 2, 2, size(O[1],4))
    O[N] = reshape(O[N][:, :, :, 3], size(O[N],1), 2, 2, 1)

    return O
end