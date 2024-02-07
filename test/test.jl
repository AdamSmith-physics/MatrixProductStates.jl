using Revise
using MatrixProductStates
using BenchmarkTools

psi = MPS(10)

println(psi)
psi[1]

MatrixProductStates.moveright!(psi)
println(psi)
MatrixProductStates.moveleft!(psi)
println(psi)

MatrixProductStates.movecentre(psi, 5)
println(psi)
psi[3]

normalise(psi)

println(psi[1])


random_psi = randomMPS(20, 2, 50, 50, 0.0)
normalise(random_psi)
println(random_psi)
@benchmark overlap(random_psi, random_psi)

@time normalise(random_psi)
println(random_psi)

overlap(random_psi, random_psi)