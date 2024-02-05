module MatrixProductStates

using LinearAlgebra: norm, svd

include("contract.jl")

include("MPS/AbstractMPS.jl")
include("MPS/MPS.jl")
include("MPS/svd.jl")
include("MPS/movecentre.jl")

end # module MatrixProductStates
``