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


# Format data to match `simulate_model` output
X = [[[[@subset(data, :Time .== t, :Repeat .== rep, :Quenched .== qu, :Temp .== 37.0)[:,ch] 
        for ch in [:Signal_Cy5,:Signal_BDP]] 
        for qu in [true,false]] 
      for rep in [1,2,3]]
        for t  in T]

# Extract the first N cells for analysis
for t in 1:length(T)
    for r in 1:3
        for q in 1:2
            for ch in 1:2
                X[t][r][q][ch] = X[t][r][q][ch][1:N]
            end
        end
    end
end
discrepancy = make_discrepancyH(X) # Make Discrepancy on all wells
pmin = 0.0
n = 1000
Rtrial = 20
ABCHybridSMC(log_hyper_prior,sample_hyper_prior,log_q,q,hyper_model,discrepancy,X,n,Rtrial,c,pmin,a,"results_par/Hierachical_real_data_intern_$M")
