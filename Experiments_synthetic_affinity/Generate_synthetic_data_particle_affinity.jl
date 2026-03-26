using JLD2
using StatsPlots
using CSV, DataFrames, DataFramesMeta
using Distributions 
using LinearAlgebra
using Statistics

using SMC
using Dist

include("functions_model_2.jl")
#M = 3       # number of wells
N = 20000    # number of cells 

################################################################
# control data for approximation
################################################################
Dcell = CSV.read("Data/THP-1/THP-1 - Control - 2015-08-05.fcs.csv",DataFrame)
Dparticle = CSV.read("Data/THP-1/110 nm CS (AF647N3).fcs.csv",DataFrame)

# channel 7 = 638 Red (Peak)
Vref = 600 # = Vparticle 
Vcell = 500 
Vexp = 500
Dcell = Dcell[:,7]*exp(0.016*(Vref - Vcell)) # Dcell <- Dcell*exp(0.016(Vref - Vcell))
Dparticle = Dparticle[:,7]

# assign time step vectors - same time step values as Alex's data set
T = 60.0*60.0*[1.0,
2.0,
4.0,
8.0,
16.0,
24.0]

# true params
θ_true = zeros(M+4)

#12:56:12 From Ryan Murphy to Everyone:
#	μr=3.86125e-7;
#	σr=4.56962e-7;
#	μK=10.0;
#	σK=2.0;
#12:58:12 From Ryan Murphy to Everyone:
#	experimental
#12:58:13 From Ryan Murphy to Everyone:
#	;The 95% highest posterior density intervals are (2.8 ~ 10−7, 5.91 ~ 10−7) for mr,
#	(2.7~10−6, 9.96~10−6) for sr, (32.1, 68.3) for mK, and (18.4, 94.7) for sK
#

θ_true = zeros(2M+6)

#         σr         σK  mr         mK    sr    sK
#λ_true = [4.56962e-7,2.0,3.86125e-7,10.0, 2e-7, 1.0]
λ_true = [4.56962e-7,2.0,3.86125e-7,10.0, 0.0, 0.0]
#λ_true = [2.92e-6,53.3,2.96e-7,51.2, 0.0, 0.0]
θ_true[2M+1:2M+6] = λ_true
mr_true = θ_true[2M+3]
mK_true = θ_true[2M+4]
sr_true = θ_true[2M+5]
sK_true = θ_true[2M+6]
θ_true[1:M] = rand(TruncatedNormal(mr_true,sr_true,0,Inf),M)   
θ_true[M+1:2M] = rand(TruncatedNormal(mK_true,sK_true,0,Inf),M)   


# make a copy of params for corrupted synthetic data
θ_true_corrupted = deepcopy(θ_true)
θ_true_corrupted2 = deepcopy(θ_true)

#corrupt well 1's association rate
θ_true_corrupted[1] = 9e-7

#corrupt well 1's association rate and well 2's carrying capacity
θ_true_corrupted2[1] = 9e-7
θ_true_corrupted2[M+2] = 20.0


# generate synthetic data
X_synth_uncorrupted = hyper_model(θ_true)
X_synth_corrupted = hyper_model(θ_true_corrupted)
X_synth_corrupted2 = hyper_model(θ_true_corrupted2)

# save synthetic data
@save "Synthetic_data_particle_affinity_v3_pooled_M$M.jld2" X_synth_corrupted X_synth_corrupted2 X_synth_uncorrupted θ_true_corrupted θ_true_corrupted2 θ_true Dcell Dparticle
