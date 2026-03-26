using JLD2
using StatsPlots
using CSV, DataFrames, DataFramesMeta
using Distributions 
using LinearAlgebra
using Statistics

using SMC
using Dist

include("functions.jl")
#M = 6
N = 1000
# ================================================================
##############################################################
## Data
##############################################################

# Load data
data = CSV.read("Data/CSV/DualMarker.csv",DataFrame)


hyperprior_dist = product_distribution(
    Uniform(0.0    , 0.3   ),   # 1.  σλ
    Uniform(0.0    , 0.1   ),   # 2.  σβ  
    Uniform(0.0    , 1.0    ),   # 3.  σR
    Uniform(0.06    , 0.2    ),   # 4.  mλ
    Uniform(0.03    , 0.1    ),   # 5.  mβ
    Uniform(0.0    , 1.0    ),   # 6.  mR
    halfCauchy(0.0 , 0.01    ),   # 7.  sλ
    halfCauchy(0.0 , 0.008    ),   # 8.  sβ   
    halfCauchy(0.0 , 0.27    ),   # 9.  sR
    Uniform(0.0    , 10.0   ),   # 10. σ₁
    Uniform(0.0    , 3.0    ),   # 11. σ₂
    Uniform(7000.0 , 10000.0),   # 12. α₁  
    Uniform(30.0   , 45.0   ),   # 13. α₂    
    Uniform(0.0    , 0.1   ),    # 14. p
    Uniform(-2.0  , 1.0    ),  # 15.  ωλ
    Uniform(-2.0  , 1.0    )  # 16.  ωβ
)


# Autofluorescence distribution
E = [@subset(data,:Time .== 0.0, :Quenched .== false, :Temp .== 37.0).Signal_Cy5,
     @subset(data,:Time .== 0.0, :Quenched .== false, :Temp .== 37.0).Signal_BDP]

# Observation times
T = sort(unique(data.Time))[2:end]

# Obtain quenching efficiency
η = 1 - (@df @subset(data,:Time .== 120.0,:Temp .== 4.0,:Quenched) mean(:Signal_Cy5)) / 
        (@df @subset(data,:Time .== 120.0,:Temp .== 4.0,(!).(:Quenched)) mean(:Signal_Cy5))



# load uncorrupted synthetic data
@load "Synthetic_data_v2_M$M.jld2" X_synth_uncorrupted
X_synth_uncorrupted = X_synth_uncorrupted

discrepancy = make_discrepancyH(X_synth_uncorrupted) # Make Discrepancy on all wells
pmin = 0.0
n = 10000
Rtrial = 20
ABCHybridSMC(log_hyper_prior,sample_hyper_prior,log_q,q,hyper_model,discrepancy,X_synth_uncorrupted,n,Rtrial,c,pmin,a,"results_par/Hierachical_synth_uncorrupted_M$M")
