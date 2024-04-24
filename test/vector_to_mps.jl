using Revise
using MatrixProductStates



psi = randomMPS(10, 2, 50, 50, 0.0)
normalise!(psi)

psi_vector = flatten(psi)

psi_mps = vector2MPS(psi_vector, 2, 0, 0.0)

psi2 = flatten(psi_mps)

psi_vector ≈ psi2

invert!(psi_mps)
psi3 = flatten(psi_mps)
psi_vector ≈ psi3

invert!(psi_mps)
psi4 = flatten(psi_mps)
psi_vector ≈ psi4