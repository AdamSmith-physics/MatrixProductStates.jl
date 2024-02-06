using Revise
using MatrixProductStates

psi = MPS(10)

println(psi)
psi[1]

MatrixProductStates.moveright!(psi)
println(psi)
MatrixProductStates.moveleft!(psi)

MatrixProductStates.movecentre(psi, 5)
println(psi)
psi[3]