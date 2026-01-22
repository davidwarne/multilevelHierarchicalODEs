#=
# Summary: Code for plotting results for the internalisation model.
#
# Author: David J. Warne (david.warne@qut.edu.au)
#         School of Mathematical Sciences
#         Queensland University of Technology
#
=#
using LaTeXStrings
using JLD2
using StatsBase
using StatsPlots
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

include("../Experiments/functions.jl")

M = 3

# load simulated synthetic data 
data_file = "./Synthetic_data_v2_M3.jld2"

# posterior samples for pooled analysis of data Scenarios  1 to 2
param_file_pu = "./results_par/Pooled_synth_uncorrupted_M3_err_25_iter_25.jld2"
param_file_pc = "./results_par/Pooled_synth_corrupted_M3_err_37_iter_21.jld2"

# posterior samples for hierarchical analysis of data Scenarios  1 to 2
param_file_u = "./results_par/Hierachical_synth_uncorrupted_M3_err_90_iter_33.jld2"
param_file_c = "./results_par/Hierachical_synth_corrupted_M3_err_120_iter_36.jld2"

# Labels for plotting
param_labels = ["\$\\mu_{\\lambda,1}\$","\$\\mu_{\\lambda,2}\$","\$\\mu_{\\lambda,3}\$",
                "\$\\mu_{\\beta,1}\$","\$\\mu_{\\beta,2}\$","\$\\mu_{\\beta,3}\$",
                "\$\\mu_{R,1}\$","\$\\mu_{R,2}\$","\$\\mu_{R,3}\$",
                "\$\\sigma_\\lambda\$","\$\\sigma_\\beta\$","\$\\sigma_R\$",
                "\$m_\\lambda\$","\$m_\\beta\$","\$m_R\$",
                "\$s_\\lambda\$","\$s_\\beta\$","\$s_R\$",
                "\$\\sigma_1\$","\$\\sigma_2\$","\$\\alpha_1\$","\$\\alpha_2\$",
                "\$p\$","\$\\omega_\\lambda\$","\$\\omega_\\beta\$"]

param_labels2 = ["\$p(\\mu_{\\lambda,1} | \\mathcal{D})\$","\$p(\\mu_{\\lambda,2} | \\mathcal{D})\$",
                 "\$p(\\mu_{\\lambda,3} | \\mathcal{D})\$","\$p(\\mu_{\\beta,1} | \\mathcal{D})\$",
                 "\$p(\\mu_{\\beta,2} | \\mathcal{D})\$","\$p(\\mu_{\\beta,3} | \\mathcal{D})\$",
                 "\$p(\\mu_{R,1} | \\mathcal{D})\$","\$p(\\mu_{R,2} | \\mathcal{D})\$",
                 "\$p(\\mu_{R,3} | \\mathcal{D})\$","\$p(\\sigma_\\lambda | \\mathcal{D})\$",
                 "\$p(\\sigma_\\beta | \\mathcal{D})\$","\$p(\\sigma_R | \\mathcal{D})\$",
                 "\$p(m_\\lambda | \\mathcal{D})\$","\$p(m_\\beta | \\mathcal{D})\$",
                 "\$p(m_R | \\mathcal{D})\$","\$p(s_\\lambda | \\mathcal{D})\$",
                "\$p(s_\\beta | \\mathcal{D})\$","\$p(s_R | \\mathcal{D})\$",
                "\$p(\\sigma_1 | \\mathcal{D})\$","\$p(\\sigma_2 | \\mathcal{D})\$",
                "\$p(\\alpha_1 | \\mathcal{D})\$","\$p(\\alpha_2 | \\mathcal{D})\$",
                "\$p(p | \\mathcal{D})\$","\$p(\\omega_\\lambda | \\mathcal{D})\$","\$p(\\omega_\\beta | \\mathcal{D})\$"]

pooled_labels = ["\$\\sigma_\\lambda\$","\$\\sigma_\\beta\$","\$\\sigma_R\$",
                "\$\\mu_\\lambda\$","\$\\mu_\\beta\$","\$\\mu_R\$",
                "\$\\sigma_1\$","\$\\sigma_2\$","\$\\alpha_1\$","\$\\alpha_2\$",
                "\$p\$","\$\\omega_\\lambda\$","\$\\omega_\\beta\$"]

pooled_labels2 = ["\$p(\\sigma_\\lambda | \\mathcal{D})\$","\$p(\\sigma_\\beta | \\mathcal{D})\$",
                 "\$p(\\sigma_R | \\mathcal{D})\$","\$p(\\mu_\\lambda | \\mathcal{D})\$",
                 "\$p(\\mu_\\beta | \\mathcal{D})\$","\$p(\\mu_R | \\mathcal{D})\$",
                 "\$p(\\sigma_1 | \\mathcal{D})\$","\$p(\\sigma_2 | \\mathcal{D})\$",
                 "\$p(\\alpha_1 | \\mathcal{D})\$","\$p(\\alpha_2 | \\mathcal{D})\$",
                 "\$p(p | \\mathcal{D})\$","\$p(\\omega_\\lambda | \\mathcal{D})\$",
                 "\$p(\\omega_\\beta | \\mathcal{D})\$"]
                
het_labels = ["\$\\lambda\$","\$\\beta\$","\$R\$"]
het_labels2 = ["\$p(\\lambda | \\mathcal{D})\$","\$p(\\beta | \\mathcal{D})\$","\$p(R | \\mathcal{D})\$"]
col = ["lightblue","orange","lightgreen"]

# get data properties
data = CSV.read("Data/CSV/DualMarker.csv",DataFrame)
# Autofluorescence distribution
E = [@subset(data,:Time .== 0.0, :Quenched .== false, :Temp .== 37.0).Signal_Cy5,
     @subset(data,:Time .== 0.0, :Quenched .== false, :Temp .== 37.0).Signal_BDP]
# Observation times
T = sort(unique(data.Time))[2:end]
# Obtain quenching efficiency
η = 1 - (@df @subset(data,:Time .== 120.0,:Temp .== 4.0,:Quenched) mean(:Signal_Cy5)) / 
        (@df @subset(data,:Time .== 120.0,:Temp .== 4.0,(!).(:Quenched)) mean(:Signal_Cy5))

# load and plot syntetic data
@load data_file X_synth_corrupted X_synth_corrupted2 X_synth_uncorrupted θ_true_corrupted θ_true_corrupted2 θ_true

# plot synthetic data (Scenario 1)
f_data_u = figure()
X = X_synth_uncorrupted
for m in 1:M
    for ti in 1:7
        subplot(M,length(T),(m-1)*length(T) + 1+(ti-1))
        if m == 1
            title(latexstring("\$t = "*string(T[ti])*"\$ [min]"))
        end
        his=PyPlot.hist(X[ti][m][1][1]/10000.0,density=true,bins = range(0.0,3.0,length=21),color="orange",edgecolor="orange",alpha=0.2)
        his=PyPlot.hist(X[ti][m][2][1]/10000.0,density=true,bins = range(0.0,3.0,length=21),color="red",edgecolor="red",alpha=0.2)
        if ti == 1
            ylabel(latexstring("Density (Rep \$"*string(m)*"\$)"))
        end
        xlim(0,3)
        if m == M
            xlabel(L"FIP-Cy5 ($10^4$)")
        end
    end
end 
display(f_data_u)
f_data_u.set_tight_layout(true)
f_data_u.set_figwidth(19.2/2.0)
f_data_u.set_figheight(10.8/2.0)


# plot synthetic data (Scenario 2)
f_data_c = figure()
X = X_synth_corrupted
for m in 1:M
    for ti in 1:7
        subplot(M,length(T),(m-1)*length(T) + 1+(ti-1))
        if m == 1
            title(latexstring("\$t = "*string(T[ti])*"\$ [min]"))
        end
        kx = range(0.0,3.0,length= 100)
        kd1 = kde(X[ti][m][1][1]/10000.0)
        kd2 = kde(X[ti][m][2][1]/10000.0)
        his=PyPlot.hist(X[ti][m][1][1]/10000.0,density=true,bins = range(0.0,3.0,length=21),color="orange",edgecolor="orange",alpha=0.2)
        his=PyPlot.hist(X[ti][m][2][1]/10000.0,density=true,bins = range(0.0,3.0,length=21),color="red",edgecolor="red",alpha=0.2)
        if ti == 1 && m == 1
            legend(["quenched","not quenched"])
        end
        if ti == 1
            ylabel(latexstring("Density (Rep \$"*string(m)*"\$)"))
        end
        xlim(0,3)
        if m == M
            xlabel(L"FIP-Cy5 ($10^4$)")
        end
    end
end 
display(f_data_c)
f_data_c.set_tight_layout(true)
f_data_c.set_figwidth(19.2/2.0)
f_data_c.set_figheight(10.8/2.0)


# load posterior samples for hierarchical analysis of each scenario
@load param_file_u θt ρₙ
θt_u = θt
ρ_u = ρₙ
@load param_file_c θt ρₙ
θt_c = θt
ρ_c= ρₙ
# set up figure dimesions
L = 5
W = 5
f_param_inf = figure()
ind = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25]
lower = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,7500.0,35.0,0.0,-2.0,-2.0]
upper = [0.1,0.1,0.1,0.1,0.1,0.1,1.0,1.0,1.0,0.02,0.1,0.3,0.15,0.1,1.0,0.1,0.05,0.5,10.0,3.0,10000.0,50.0,0.1,1.0,1.0]
for k in 1:length(ind)
    i = ind[k]
    subplot(L,W,k)
    kxu = range(lower[i],upper[i],length=500)
    kdu = kde(θt_u[i,:])
    kdc = kde(θt_c[i,:])
    his = PyPlot.plot(kxu,pdf(kdu,kxu), color="lightblue")
    his = PyPlot.plot(kxu,pdf(kdc,kxu), color="orange",linestyle="dashed")
    his = PyPlot.plot([θ_true[i],θ_true[i]],[0,maximum(pdf(kdu,kxu))], linestyle="--",color="darkblue")
    if i == 1
        his = PyPlot.plot([θ_true_corrupted[i],θ_true_corrupted[i]],[0,maximum(pdf(kdc,kxu))], linestyle="--",color="red")
    end
    his = PyPlot.fill_between(kxu,pdf(kdu,kxu), color="lightblue",alpha=0.2)
    his = PyPlot.fill_between(kxu,pdf(kdc,kxu), color="orange",alpha=0.2)
    if i==1
        legend(["Scenario 1","Scenario 2"],loc="upper right")
    end
    xlabel(latexstring(param_labels[i]))
    ylabel(latexstring(param_labels2[i]))
end
display(f_param_inf)
f_param_inf.set_tight_layout(true)

# load posterior samples for pooled analysis of each scenario
@load param_file_pu θt ρₙ
θt_pu = θt
ρ_pu = ρₙ
@load param_file_pc θt ρₙ
θt_pc = θt
ρ_pc= ρₙ
# set up figure dimesions
L = 1
W = 3
f_param_inf_p = figure()
ind = [4, 1, 11]
lower = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,7500.0,35.0,0.0,-2.0,-2.0]
upper = [0.12,0.12,0.3,0.1,0.1,1.0,10.0,3.0,10000.0,50.0,0.1,1.0,1.0]
for k in 1:length(ind)
    i = ind[k]
    subplot(L,W,k)
    kxu = range(lower[i],upper[i],length=500)
    kdu = kde(θt_pu[i,:])
    kdc = kde(θt_pc[i,:])
    his = PyPlot.plot(kxu,pdf(kdu,kxu), color="lightblue")
    his = PyPlot.plot(kxu,pdf(kdc,kxu), color="orange",linestyle="dashed")
    his = PyPlot.fill_between(kxu,pdf(kdu,kxu), color="lightblue",alpha=0.2)
    his = PyPlot.fill_between(kxu,pdf(kdc,kxu), color="orange",alpha=0.2)
    if k==1
        legend(["Scenario 1","Scenario 2"])
    end
    xlabel(latexstring(pooled_labels[i]))
    ylabel(latexstring(pooled_labels2[i]))
end
display(f_param_inf_p)
f_param_inf_p.set_tight_layout(true)

# comparison of cell parameters under pooled and hierarchical analuyses
θ_scen_p = [θt_pu, θt_pc]
θ_scen_h = [θt_u, θt_c]

lower = [0.0,0.0,0.0]
upper = [0.3,0.2,3.0]
indμ = [4,5,6]
indσ = [1,2,3]
indω = [12,13,0]
L = 1
W = 3
scen = 2
f_het_uncert = figure() 
for i = 1:length(indμ)
    x = range(lower[i],upper[i],length=500)
    j = scen
    subplot(L,W,i)
    μ = θ_scen_p[j][indμ[i],:]
    σ = θ_scen_p[j][indσ[i],:]
    ω = []
    if indω[i] != 0
        ω = θ_scen_p[j][indω[i],:]
    end
    p = LogNormalAlt2(μ[1],σ[1])
    p = 0
    if indω[i] !=0 
        p = GammaAlt(μ[1],σ[1],ω[1])
    else
        p = LogNormalAlt(1.0,μ[1],σ[1])
    end
    pdfsum = pdf.(p,x)
    for k =2:length(μ)
        p = 0
        if indω[i] !=0 
            p = GammaAlt(μ[k],σ[k],ω[k])
        else
            p = LogNormalAlt(1.0,μ[k],σ[k])
        end
        pdfsum += pdf.(p,x)
    end
    his = PyPlot.plot(x,pdfsum/length(μ),color="black",alpha=1)
    xlabel(latexstring(het_labels[i]))
    ylabel(latexstring(het_labels2[i]))
end

indμ = [1,4,7,2,5,8,3,6,9]
indσ = [10,11,12,10,11,12,10,11,12]
indm = [13,14,15,13,14,15,13,14,15]
inds = [16,17,18,16,17,18,16,17,18]
indω = [24,25,0,24,25,0,24,25,0]
indcol = [1,2,3,1,2,3,1,2,3]
indcol2 = [1,1,1,2,2,2,3,3,3]
lower = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
upper = [0.3,0.2,3.0,0.3,0.2,3.0,0.3,0.2,3.0]
for i = 1:length(indμ)
    subplot(L,W,indcol[i])
    x = range(lower[i],upper[i],length=500)
    j = scen
    μ = θ_scen_h[j][indμ[i],:]
    σ = θ_scen_h[j][indσ[i],:]
    ω = []
    if indω[i] != 0
        ω = θ_scen_h[j][indω[i],:]
    end
    p = 0
    if indω[i] !=0 
        p = GammaAlt(μ[1],σ[1],ω[1])
    else
        p = LogNormalAlt(1.0,μ[1],σ[1])
    end
    pdfsum = pdf.(p,x)
    for k =2:length(μ)
        p = 0
        if indω[i] !=0 
            p = GammaAlt(μ[k],σ[k],ω[k])
        else
            p = LogNormalAlt(1.0,μ[k],σ[k])
        end
        pdfsum += pdf.(p,x)
    end
    his = PyPlot.plot(x,pdfsum/length(μ),color=col[indcol2[i]],alpha=1)
end

subplot(L,W,1)
legend(["Pooled sample" ,"Replicate 1", "Replicate 2","Replicate 3"])
display(f_het_uncert)
f_het_uncert.set_tight_layout(true)

