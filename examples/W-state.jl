using Revise
using MatrixProductStates
using TensorOperations
using LinearAlgebra


function create_W_state(N)

    tensors = []
    for n in 1:N

        a = sqrt((N-n)/(N-n+1))
        b = sqrt(1/(N-n+1))

        W = zeros(2, 2, 2)  # vl, p, vr
        W[:, 1, :] = [a 0; 0 1]
        W[:, 2, :] = [0 b; 0 0]

        if n == 1
            W = reshape(W[1, :, :],(1,2,2))
        elseif n == N
            W = reshape(W[:, :, 2],(2,2,1))
        end

        append!(tensors, [W])

    end

    return MPS(2, N, tensors, 1, 0, 0.0)

end


N = 3

Wstate = create_W_state(N)
#Wstate = randomMPS(N, 2, 2, 0, 0.0)
#normalise!(Wstate)

flatten(Wstate)
norm(flatten(Wstate))

centre = 2

MatrixProductStates.movecentre!(Wstate, centre)
Gamma_tilde = Wstate[centre]
@tensor out[vr1, vr2] := conj(Gamma_tilde)[vl, p, vr1] * Gamma_tilde[vl, p, vr2]

Gamma_tilde = reshape(Gamma_tilde, (2, 4))

F = svd(Gamma_tilde)
U = F.U
Sl = F.S

Gamma_tilde = reshape(Gamma_tilde, (4, 2))

F = svd(Gamma_tilde)
Sr = F.S
Vt = F.Vt

@tensor Gamma[vl, p, vr] := diagm(Sl.^(-1))[vl, c1] * Wstate[centre][c1, p, c2] * diagm(Sr.^(-1))[c2, vr]

@tensor out[p1,p2] := conj(Gamma)[vl, p1, vr] * Gamma[vl, p2, vr]

@tensor out[vr1, vr2] := conj(Gamma)[vl, p, vr1] * Gamma[vl, p, vr2]
@tensor out[vr1, vr2] := conj(Gamma)[vl, p, vr1] * Gamma[vl, p, vr2]

Gamma_matrix = permutedims(Gamma, (1, 3, 2))
Gamma_matrix = reshape(Gamma_matrix, (2, 4))




MatrixProductStates.movecentre!(Wstate, 1)
B = Wstate[centre]
@tensor out[vl1, vl2] := conj(B)[vl1, p, vr] * B[vl2, p, vr]

B = reshape(B, (4, 2))

F = svd(B)
Sl = F.S




U = reshape(U, (2, 2, 2))
U[:,2,:]

@tensor Gamma[vl, p, vr] := diagm(S.^(-1))[vl, c1] * Wstate[2][c1, p, c2] * diagm(S.^(-1))[c2, vr]

Gamma[:, 1, :]
Gamma[:, 2, :]

@tensor out[p1,p2] := conj(Gamma)[vl, p1, vr] * Gamma[vl, p2, vr]

@tensor out[vr1, vr2] := conj(Gamma)[vl, p, vr1] * Gamma[vl, p, vr2]
@tensor out[vr1, vr2] := conj(Gamma)[vl, p, vr1] * Gamma[vl, p, vr2]


MatrixProductStates.movecentre!(Wstate, 2)

Wstate.tensors[1][:, 1, :]
Wstate.tensors[1][:, 2, :]

Wstate.tensors[2][:, 1, :]
Wstate.tensors[2][:, 2, :]

Wstate.tensors[3][:, 1, :]
Wstate.tensors[3][:, 2, :]

flatten(Wstate)