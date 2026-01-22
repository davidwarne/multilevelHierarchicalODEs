using JLD2
using StatsPlots
using CSV, DataFrames, DataFramesMeta
using Distributions 
using LinearAlgebra
using Statistics

using SMC
using Dist

include("functions_model_2.jl")
M = 3
N = 20000

# ================================================================

prior_dist = product_distribution(
     Uniform(0.0,1e-6),     # 1. σr
     Uniform(0.0,10.0),    # 2. σK
     Uniform(0.0,1e-6),     # 3. μr
     Uniform(0.0,50.0)    # 4. μK
)
# Observation times
# assign time step vectors - same time step values as Ryan's data set
T = 60.0*60.0*[1.0,
2.0,
4.0,
8.0,
16.0,
24.0]

# load corrupted synthetic data
@load "Synthetic_data_particle_affinity_v3_pooled_M$M.jld2" X_synth_corrupted2 Dcell Dparticle
X_synth_corrupted2 = X_synth_corrupted2

# create a new 2 level vector with a size of 7-elements (size of time steps)
reps = Vector{Vector{Float64}}(undef, length(T))

# flatten the 3 level vector to a 2 level vector
function flatten_vector(X) 
        # iterate through each time step
        for i in 1:size(X)[1]
                d = deepcopy(X)
                v1 = d[i][1]
                for j = 2:M
                    v1 = vcat(v1,d[i][j]) 
                end 
                
                reps[i] = v1

        end
        return reps 
end


# flatten the synthetic corrupted data so that all 3 wells are in one vector
flatten_X_synth_corrupted2 = flatten_vector(X_synth_corrupted2)

discrepancy = make_discrepancy(flatten_X_synth_corrupted2) # Make Discrepancy on all wells
pmin = 0.0
n = 2000
Rtrial = 50
a = 0.5
ABCHybridSMC(log_prior,sample_prior,log_q,q,model,discrepancy,flatten_X_synth_corrupted2,n,Rtrial,c,pmin,a,"results_par2/Pooled_v2_synth_corrupted2_particle_affinity_M$M")
