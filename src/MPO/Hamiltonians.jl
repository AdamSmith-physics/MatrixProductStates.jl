export TFIM
function TFIM(N::Int, J::Real, g::Real, h::Real)::MPO

    Id = [1.0 0.0; 0.0 1.0]
    X = [0.0 1.0; 1.0 0.0]
    Y = [0.0 -im; im 0.0]
    Z = [1.0 0.0; 0.0 -1.0]

    O = MPO(N, 2)
    W = zeros(ComplexF64, 3, 2, 2, 3)
    W[1, :, :, 1] = Id
    W[1, :, :, 2] = Z
    W[1, :, :, 3] = -J*(h*Z + g*X)
    W[2, :, :, 3] = -J*Z
    W[3, :, :, 3] = Id

    for i in 1:N
        O.tensors[i] = copy(W)
    end
    O[1] = reshape(O[1][1, :, :, :], 1, 2, 2, size(O[1],4))
    O[N] = reshape(O[N][:, :, :, 3], size(O[N],1), 2, 2, 1)

    return O
end