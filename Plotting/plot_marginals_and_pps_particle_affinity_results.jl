#=
# Summary: Code for plotting results for the particle-cell interaction model.
#
# Author: David J. Warne (david.warne@qut.edu.au)
#         School of Mathematical Sciences
#         Queensland University of Technology
#
=#
using LaTeXStrings
using JLD2
using StatsBase
using CSV
using DataFrames, DataFramesMeta
using PyPlot
using Random
using Distributions
using KernelDensity
using LinearAlgebra
using QuadGK
using SMC

rc("font",family="serif")
rc("text",usetex="True")

include("../Experiments_affinity/functions_model_2.jl")

M = 3

# load simulated synthetic data 
data_file = "./Synthetic_data_particle_affinity_v3_M3.jld2"

# posterior samples for pooled analysis of data Scenarios  1 to 3
param_file_pu = "./results_final_particle_affinity/Pooled_v2_synth_uncorrupted_particle_affinity_M3_err_18_iter_16.jld2"
param_file_pc = "./results_final_particle_affinity/Pooled_v2_synth_corrupted_particle_affinity_M3_err_16_iter_16.jld2"
param_file_pc2 = "./results_final_particle_affinity/Pooled_v2_synth_corrupted2_particle_affinity_M3_err_26_iter_15.jld2"

# posterior samples for hierarchical analysis of data Scenarios  1 to 3
param_file_u = "./results_final_particle_affinity/Hierachical_synth_uncorrupted_particle_affinity_M3_err_61_iter_48.jld2"
param_file_c = "./results_final_particle_affinity/Hierachical_synth_corrupted_particle_affinity_M3_err_86_iter_33.jld2"
param_file_c2 = "./results_final_particle_affinity/Hierachical_synth_corrupted2_particle_affinity_M3_err_262_iter_22.jld2"

# Labels for plotting
param_labels = ["\$\\mu_{r,1}\$", "\$\\mu_{r,2}\$","\$\\mu_{r,3}\$","\$\\mu_{K,1}\$",
                "\$\\mu_{K,2}\$","\$\\mu_{K,3}\$","\$\\sigma_r\$","\$\\sigma_K\$",
                "\$m_r\$","\$m_K\$","\$s_r\$", "\$s_K\$"]
param_labels2 = ["\$p(\\mu_{r,1} | \\mathcal{D})\$","\$p(\\mu_{r,2} | \\mathcal{D})\$",
                 "\$p(\\mu_{r,3} | \\mathcal{D})\$","\$p(\\mu_{K,1} | \\mathcal{D})\$",
                 "\$p(\\mu_{K,2} | \\mathcal{D})\$","\$p(\\mu_{K,3} | \\mathcal{D})\$",
                 "\$p(\\sigma_r | \\mathcal{D})\$","\$p(\\sigma_K | \\mathcal{D})\$",
                 "\$p(m_r | \\mathcal{D})\$","\$p(m_K | \\mathcal{D})\$",
                 "\$p(s_r | \\mathcal{D})\$","\$p(s_K | \\mathcal{D})\$"]

pooled_labels = ["\$\\sigma_r\$","\$\\sigma_K\$","\$\\mu_r\$","\$\\mu_K\$"]

pooled_labels2 = ["\$p(\\sigma_r | \\mathcal{D})\$","\$p(\\sigma_K | \\mathcal{D})\$",
                 "\$p(\\mu_r | \\mathcal{D})\$","\$p(\\mu_K | \\mathcal{D})\$"]

het_labels = ["\$r\$","\$K\$"]
het_labels2 = ["\$p(r | \\mathcal{D})\$","\$p(K | \\mathcal{D})\$"]
# assign time step vectors - same time step values as Ryan's data set
T = 60.0*60.0*[1.0,2.0,4.0,8.0,16.0,24.0]

# load and plot syntetic data
@load data_file X_synth_corrupted X_synth_corrupted2 X_synth_uncorrupted θ_true_corrupted θ_true_corrupted2 θ_true

# plot synthetic data (Scenario 1)
f_data_u = figure()
X = X_synth_uncorrupted
for m in 1:M
    for ti in 1:length(T)
        subplot(M,length(T),(m-1)*length(T) + 1+(ti-1))
        if m == 1
            title(latexstring("\$t = "*string(T[ti]/3600.0)*"\$ [hr]"))
        end
        his=PyPlot.hist(X[ti][m],density=true,bins = 0:50:1000,color="lightblue",edgecolor="blue")
        xlim(0,1000)
        if ti == 1
            ylabel(latexstring("Density (Rep \$"*string(m)*"\$)"))
        end
        if m == M
            xlabel("S [AU]")
        end
    end
end 

# plot synthetic data (Scenario 2)
f_data_c = figure()
X = X_synth_corrupted
for m in 1:M
    for ti in 1:length(T)
        subplot(M,length(T),(m-1)*length(T) + 1+(ti-1))
        if m == 1
            title(latexstring("\$t = "*string(T[ti]/3600.0)*"\$ [hr]"))
        end
        if m == 1
            his=PyPlot.hist(X[ti][m],density=true,bins = 0:50:1000,color="orange",edgecolor="red")
        else
            his=PyPlot.hist(X[ti][m],density=true,bins = 0:50:1000,color="lightblue",edgecolor="blue")
        end 
        xlim(0,1000)
        if ti == 1
            ylabel(latexstring("Density (Rep \$"*string(m)*"\$)"))
        end
        if m == M
            xlabel("S [AU]")
        end
    end
end 

# plot synthetic data (Scenario 2)
f_data_c2 = figure()
X = X_synth_corrupted2
for m in 1:M
    for ti in 1:length(T)
        subplot(M,length(T),(m-1)*length(T) + 1+(ti-1))
        if m == 1
            title(latexstring("\$t = "*string(T[ti]/3600.0)*"\$ [hr]"))
        end
        if m == 1
            his=PyPlot.hist(X[ti][m],density=true,bins = 0:50:1000,color="orange",edgecolor="red")
        elseif m == 2
            his=PyPlot.hist(X[ti][m],density=true,bins = 0:50:1000,color="lightgreen",edgecolor="darkgreen")
        else 
            his=PyPlot.hist(X[ti][m],density=true,bins = 0:50:1000,color="lightblue",edgecolor="blue")
        end
        xlim(0,1000)
        if ti == 1
            ylabel(latexstring("Density (Rep \$"*string(m)*"\$)"))
        end
        if m == M
            xlabel("S [AU]")
        end
    end
end 

# load posterior samples for hierarchical analysis of each scenario
@load param_file_u θt ρₙ
θt_u = θt
ρ_u = ρₙ
@load param_file_c θt ρₙ
θt_c = θt
ρ_c= ρₙ
@load param_file_c2 θt ρₙ
θt_c2 = θt
ρ_c2 = ρₙ
# set up figure dimesions
L = 3
W = 4
f_param_inf = figure()
lower = [0.0,0.0,0.0,7.0,7.0,7.0,0.0,0.0,0.0,0.0,0.0,0.0]
upper = [1.5e-6,1.5e-6,1.5e-6,25.0,25.0,25.0,1e-6,5.0,1.5e-6,25.0,1e-6,20.0]
for k in 1:length(θ_true)
    ind = [1 2 3 7 4 5 6 8 9 10 11 12]
    i = ind[k]
    subplot(L,W,k)
    kxu = range(lower[i],upper[i],length=500)
    # get density estimates
    kdu = kde(θt_u[i,:])
    kdc = kde(θt_c[i,:])
    kdc2 = kde(θt_c2[i,:])
    # plot densities
    his = PyPlot.plot(kxu,pdf(kdu,kxu), color="lightblue")
    his = PyPlot.plot(kxu,pdf(kdc,kxu), color="orange",linestyle="dashed")
    his = PyPlot.plot(kxu,pdf(kdc2,kxu), color="lightgreen", linestyle="dotted")
    his = PyPlot.fill_between(kxu,pdf(kdu,kxu), color="lightblue",alpha=0.2)
    his = PyPlot.fill_between(kxu,pdf(kdc,kxu), color="orange",alpha=0.2)
    his = PyPlot.fill_between(kxu,pdf(kdc2,kxu), color="lightgreen",alpha=0.2)
    if i==1
        legend(["Scenario 1","Scenario 2","Scenario 3"])
    end
    xlabel(latexstring(param_labels[i]))
    ylabel(latexstring(param_labels2[i]))
end
display(f_param_inf)
f_param_inf.set_tight_layout(true)
f_param_inf.set_figwidth(19.2/2.0)
f_param_inf.set_figheight(10.8/2.0)

# load posterior samples for pooled analysis of each scenario
@load param_file_pu θt ρₙ
θt_pu = θt
ρ_pu = ρₙ
@load param_file_pc θt ρₙ
θt_pc = θt
ρ_pc= ρₙ
@load param_file_pc2 θt ρₙ
θt_pc2 = θt
ρ_pc2= ρₙ
# set up figure dimensions
L = 2
W = 2
f_param_inf_p = figure()
lower = [0.0,0.0,0.0,8.0]
upper = [2.0e-6,10.0,1e-6,15.0]
ind = [3 4 1 2]
for k in 1:length(ind)
    i = ind[k]
    subplot(L,W,k)
    kxu = range(lower[i],upper[i],length=500)
    # get density estimates
    kdu = kde(θt_pu[i,:])
    kdc = kde(θt_pc[i,:])
    kdc2 = kde(θt_pc2[i,:])
    # plot densities
    his = PyPlot.plot(kxu,pdf(kdu,kxu), color="lightblue")
    his = PyPlot.plot(kxu,pdf(kdc,kxu), color="orange",linestyle="dashed")
    his = PyPlot.plot(kxu,pdf(kdc2,kxu), color="lightgreen",linestyle="dotted")
    his = PyPlot.fill_between(kxu,pdf(kdu,kxu), color="lightblue",alpha=0.2)
    his = PyPlot.fill_between(kxu,pdf(kdc,kxu), color="orange",alpha=0.2)
    his = PyPlot.fill_between(kxu,pdf(kdc2,kxu), color="lightgreen",alpha=0.2)
    if k==1
        legend(["Scenario 1","Scenario 2","Scenario 3"],loc="upper left")
    end
    xlabel(latexstring(pooled_labels[i]))
    ylabel(latexstring(pooled_labels2[i]))
end
display(f_param_inf_p)
f_param_inf_p.set_tight_layout(true)


# comparison of cell parameters under pooled and hierarchical analuyses
θ_scen_p = [θt_pu, θt_pc, θt_pc2]
θ_scen_h = [θt_u, θt_c, θt_c2]

lower = [0.0,0.0]
upper = [2e-6,30.0]
indμ = [3,4]
indσ = [1,2]

scen = 3
L = 1
W = 2
f_het_uncert = figure() 
col = ["lightblue","orange","lightgreen"]
for i = 1:length(indμ)
    x = range(lower[i],upper[i],length=100)
    j = scen
    subplot(L,W,i)
    μ = θ_scen_p[j][indμ[i],:]
    σ = θ_scen_p[j][indσ[i],:]
    p = LogNormalAlt2(μ[1],σ[1])
    pdfsum = pdf.(p,x)
    for k =2:length(μ)
        p = LogNormalAlt2(μ[k],σ[k])
        pdfsum += pdf.(p,x)
    end
    his = PyPlot.plot(x,pdfsum/length(μ),color="black",alpha=1)
    xlabel(latexstring(het_labels[i]))
    ylabel(latexstring(het_labels2[i]))
    
end

indμ = [1,4,2,5,3,6]
indσ = [7,8,7,8,7,8]
indm = [9,10,9,10,9,10]
inds = [11,12,11,12,11,12]
indcol = [1,2,1,2,1,2]
indcol2 = [1,1,2,2,3,3]
lower = [0.0,0.0,0.0,0.0,0.0,0.0]
upper = [2e-6,30.0,2e-6,30.0,2e-6,30.0]
for i = 1:length(indμ)
    x = range(lower[i],upper[i],length=100)
    j = scen    
    subplot(L,W,indcol[i])
    μ = θ_scen_h[j][indμ[i],:]
    σ = θ_scen_h[j][indσ[i],:]
    p = LogNormalAlt2(μ[1],σ[1])
    pdfsum = pdf.(p,x)
    for k =2:length(μ)
        p = LogNormalAlt2(μ[k],σ[k])
        pdfsum += pdf.(p,x)
    end
    his = PyPlot.plot(x,pdfsum/length(μ),color=col[indcol2[i]],alpha=1)
end

subplot(L,W,1)
legend(["Pooled sample" ,"Replicate 1", "Replicate 2","Replicate 3"])


