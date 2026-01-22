using JLD2
using StatsPlots
using CSV, DataFrames, DataFramesMeta
using Distributions 
using LinearAlgebra
using Statistics

using SMC
using Dist

include("../Experiments/functions.jl")
print("got here $M")
#M = 6
N = 1000

data = CSV.read("Data/CSV/DualMarker.csv",DataFrame)
# Autofluorescence distribution
E = [@subset(data,:Time .== 0.0, :Quenched .== false, :Temp .== 37.0).Signal_Cy5,
     @subset(data,:Time .== 0.0, :Quenched .== false, :Temp .== 37.0).Signal_BDP]
# Obtain quenching efficiency
η = 1 - (@df @subset(data,:Time .== 120.0,:Temp .== 4.0,:Quenched) mean(:Signal_Cy5)) / 
        (@df @subset(data,:Time .== 120.0,:Temp .== 4.0,(!).(:Quenched)) mean(:Signal_Cy5))



T = sort(unique(data.Time))[2:end]
# true params
θ_true = zeros(3M+16)

# σλ σβ  σR mλ, mβ mR sλ sβ sR σ₁ σ₂ α₁ α₂ p ωλ ωβ

λ_true = [0.01,0.05,0.25,0.08,0.05,0.05,0.01,0.005,0.01,5.0,2.0,8500.0,38.0,0.075,0.9,-1.0] 
θ_true[3M+1:3M+16] = λ_true
mλ_true = θ_true[3M+4]    # mean of λ means
mβ_true = θ_true[3M+5]    # mean of β means
mR_true = θ_true[3M+6]
sλ_true = θ_true[3M+7]    # STD of β means
sβ_true = θ_true[3M+8]    # STD of λ means 
sR_true = θ_true[3M+9]

θ_true[1:M]     = rand(TruncatedNormal(mλ_true,sλ_true,0,Inf),M)
θ_true[M+1:2M]  = rand(TruncatedNormal(mβ_true,sβ_true,0,Inf),M)
θ_true[2M+1:3M] = rand(TruncatedNormal(mR_true,sR_true,0,Inf),M)

θ_true_corrupted = deepcopy(θ_true)
θ_true_corrupted2 = deepcopy(θ_true)

#corrupt well 1's and M's nternalisation rate
θ_true_corrupted[1] = 0.02
θ_true_corrupted2[1] = 0.02
θ_true_corrupted2[M] = 0.14

# generate synthetic data
X_synth_uncorrupted = hyper_model(θ_true)
X_synth_corrupted = hyper_model(θ_true_corrupted)
X_synth_corrupted2 = hyper_model(θ_true_corrupted2)
@save "Synthetic_data_v2_M$M.jld2" X_synth_corrupted X_synth_corrupted2 X_synth_uncorrupted θ_true_corrupted θ_true_corrupted2 θ_true 
