using JLD2
using StatsPlots
using CSV, DataFrames, DataFramesMeta
using Distributions 
using LinearAlgebra
using Statistics

using SMC
using Dist

include("functions_model_2.jl")
#M = 3

N = 20000    # number of cells 
# assign time step vectors - same time step values as Ryan's data set
T = 60.0*60.0*[1.0,
2.0,
4.0,
8.0,
16.0,
24.0]
# ================================================================

hyperprior_dist = product_distribution(
    Uniform(0.0    , 1e-6   ),     # σr  
    Uniform(0.0    , 5.0   ),     # σK  
    Uniform(0.0    , 1e-6   ),     # mr
    Uniform(0.0    , 50.0   ),     # mK
    halfCauchy(0.0    , 3e-7    ), # sr
    halfCauchy(0.0    , 0.5    ), # sK

)



# load corrupted synthetic data
@load "Synthetic_data_particle_affinity_v3_M$M.jld2" X_synth_uncorrupted Dcell Dparticle
X_synth_uncorrupted = X_synth_uncorrupted

discrepancy = make_discrepancyH(X_synth_uncorrupted) # Make Discrepancy on all wells
pmin = 0.0
n = 2000
Rtrial = 50
a = 0.5
ABCHybridSMC(log_hyper_prior,sample_hyper_prior,log_q,q,hyper_model,discrepancy,X_synth_uncorrupted,n,Rtrial,c,pmin,a,"results_par/Hierachical_synth_uncorrupted_particle_affinity_M$M")
