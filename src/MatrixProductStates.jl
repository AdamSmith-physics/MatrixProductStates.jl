module MatrixProductStates

using TensorOperations
using LinearAlgebra
import LinearAlgebra: norm, svd

include("contract.jl")

include("MPS/AbstractMPS.jl")
include("MPS/MPS.jl")
include("MPS/svd.jl")
include("MPS/movecentre.jl")
include("MPS/normalise.jl")
include("MPS/overlap.jl")

end # module MatrixProductStates
``