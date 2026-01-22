using JLD2
using StatsPlots
using CSV, DataFrames, DataFramesMeta
using Distributions 
using LinearAlgebra
using Statistics

using SMC
using Dist

include("functions.jl")
M = 3
N = 1000
# ================================================================
##############################################################
## Data
##############################################################

# Load data
data = CSV.read("Data/CSV/DualMarker.csv",DataFrame)


# Autofluorescence distribution
# filtering dataset
E = [@subset(data,:Time .== 0.0, :Quenched .== false, :Temp .== 37.0).Signal_Cy5,
     @subset(data,:Time .== 0.0, :Quenched .== false, :Temp .== 37.0).Signal_BDP]

# Observation times
T = sort(unique(data.Time))[2:end]

# Format data to match `simulate_model` output
X = [[[@subset(data, :Time .== t, :Quenched .== qu, :Temp .== 37.0)[:,ch] 
        for ch in [:Signal_Cy5,:Signal_BDP]] 
        for qu in [true,false]] 
        for t  in T]

# Obtain quenching efficiency
# data with low temperature, probablity of quenched cells being quenched
η = 1 - (@df @subset(data,:Time .== 120.0,:Temp .== 4.0,:Quenched) mean(:Signal_Cy5)) / 
        (@df @subset(data,:Time .== 120.0,:Temp .== 4.0,(!).(:Quenched)) mean(:Signal_Cy5))


# load corrupted synthetic data
@load "Synthetic_data_v2.jld2" X_synth_corrupted
X_synth_corrupted = X_synth_corrupted



# create a new 4 level vector with a size of 7-elements (size of time steps)
reps = Vector{Vector{Vector{Vector{Float64}}}}(undef, size(X_synth_corrupted)[1])

# function to flatten the 5 level vector to a 4 level vector
function flatten_vector(X) 
        # iterate through each time step
        for i in 1:size(X)[1]
                d = deepcopy(X)
                v1 = vcat(d[i][1][1][1], d[i][2][1][1], d[i][3][1][1]) 
                v2 = vcat(d[i][1][1][2], d[i][2][1][2], d[i][3][1][2])
                v3 = vcat(d[i][1][2][1], d[i][2][2][1], d[i][3][2][1])
                v4 = vcat(d[i][1][2][2], d[i][2][2][2], d[i][3][2][2])
                
                v_2_level = [[v1, v2], [v3, v4]]
                
                reps[i] = v_2_level

        end
        return reps 
end

# flatten the synthetic corrupted data so that all 3 wells are in one vector for pooled model
flatten_X_synth_corrupted = flatten_vector(X_synth_corrupted)


discrepancy = make_discrepancy(flatten_X_synth_corrupted) # Make Discrepancy on all wells
# pmin = pmin/10.0
# n = 1000
# Rtrial = 20

ABCHybridSMC(log_prior,sample_prior,log_q,q,model,discrepancy,flatten_X_synth_corrupted,n,Rtrial,c,pmin,a,"results_par/Pooled_synth_corrupted_internalisation")