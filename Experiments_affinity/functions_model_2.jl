#
#=  
# Create functions for particle-cell affinity model only, internalisation model uses `functions.jl` file
# 
# Centralised so all things use the same functions
# 
#       author: Thomas P. Steele (tp.steele00@gmail.com)
=#
using Interpolations
using LinearAlgebra

using Dist

# ================================================================
# Prior functions
# ================================================================

# assign constant values (Based on Murphy et al., 2024)
V = 1.01e-6 # volume of solution
u0 = 9.95e7 # contentration of particles-per-cell in a well mixed solution at time 0 
C = 1.0 # fractional surface coverage of cells
S = 5.31e-5 # surface area of cell boundary

log_q = (θp,θc,Σ) -> logpdf(MultivariateNormal(θc,Σ),θp)
q = (θc,Σ) -> rand(MultivariateNormal(θc,Σ))


function sample_prior()
    return rand(prior_dist)
end

function log_prior(θ)
    # extract standard deviation hyper parameters
    σr = θ[1] # STD of individual r
    σK = θ[2] # STD of individual K

    # ensure standard deviations are greater than zero
    if σr < 0 || σK < 0
        return -Inf
    else
        return logpdf(prior_dist,θ)
    end
end

log_hyperprior = (λ) -> logpdf(hyperprior_dist,λ)
function log_hyper_prior(θ)
    # we have Πⱼp(ϕⱼ| λ) p(λ) where j = 1,2,...,M (number of groups)
    # for this example αⱼⁱ~ N(μαⱼ,σα²) for i = 1,...,N and j = 1,...,M
    # extract hyper parameters

    # in group means
    μr = @view θ[1:M]     # mean of individual r per group
    μK = @view θ[M+1:2M]     # mean of individual K per group

    # hyper means and vars
    σr = θ[2M+1] #STD for r within group    
    σK = θ[2M+2] #STD for K within group   
    mr = θ[2M+3]  # mean of r across groups  
    mK = θ[2M+4]  # mean of K across groups  
    sr = θ[2M+5]  # STD of r across groups 
    sK = θ[2M+6]  # STD of K across groups 
    
    if σr < 0 || σK < 0 || mr < 0 || mK <0 || sr < 0 || sK < 0 || any(μr .< 0) || any(μK .< 0)
        log_p = -Inf
    else
        log_p = log_hyperprior([σr,σK,mr,mK,sr,sK]) + sum(logpdf.(Normal(mr, sr), μr)) + sum(logpdf.(Normal(mK, sK), μK))
    end
    return log_p
end

function sample_hyper_prior()
    θ = zeros(2M+6)
    λ = rand(hyperprior_dist);
    θ[2M+1:2M+6] = λ

    mr = θ[2M+3]    
    mK = θ[2M+4]    
    sr = θ[2M+5]    
    sK = θ[2M+6]    

    θ[1:M]     = rand(TruncatedNormal(mr,sr,0,Inf),M)
    θ[M+1:2M]  = rand(TruncatedNormal(mK,sK,0,Inf),M)
    return θ
end

# ================================================================
# Model functions for particle-cell affinity model
# ================================================================

function solve_ode_model(t::Float64, r::Float64, K::Float64) # t: time steps, r: particle-cell association rate, K: maximum particles per cell (cell carrying capacity for particles)

    
    if (K != V*u0) && (V*u0 > 0) 
        P = V*u0*(1 - (V*u0 - K)/(V*u0 - K*exp(-(r*C*S*(V*u0 - K)*t)/(K*V))))
    elseif (K == V * u0) && (V * u0 > 0) 
        P = (r*C*S*(u0^2)*V*t)/(K + r* C*S*u0*t)
    else
        P = nothing
    end 
    return P
end

# simulated individual model
function model(θ)
    # extract hyper parameters

    σr = θ[1] # STD of r for cells
    σK = θ[2] # STD of K for cells
    μr = θ[3] # mean of r for cells
    μK = θ[4] # mean of K for cells

    # forward sim in time
    results = Vector{Vector{Float64}}(undef,length(T))

    Tfine = 0:60*30:T[end]
    ū = Vector{Float64}(undef, length(Tfine))
    for (idx,t) in enumerate(Tfine)
        N_prep = 100
        p_prep = Vector{Float64}(undef,N_prep)
        # sample parameter for each cell
        r_prep  = rand(LogNormalAlt2(μr,σr),N_prep)  
        K_prep = rand(LogNormalAlt2(μK,σK),N_prep)  
        for j in 1:N_prep
            p_prep[j] = solve_ode_model(t,r_prep[j],K_prep[j])
        end
        u = u0 .- p_prep/V
        ū[idx] = sum(u)/N_prep
    end

    #use trap rule to get ∫ū(t)dt
    ∫ūdt = deepcopy(ū)
    for (idx,t) in enumerate(Tfine)
        if idx > 1
            ∫ūdt[idx] = ∫ūdt[idx-1] + 0.5*(ū[idx] + ū[idx-1])*(Tfine[idx]-Tfine[idx-1])
        else
            ∫ūdt[idx] = 0.5*(ū[idx] + u0)*(Tfine[idx])
        end
    end
    for (idx,t) in enumerate(T)
        dsim = Vector{Float64}(undef,N)
        idxfine = findfirst(Tfine .== T[idx])
        # sample parameter for each individual
        r  = rand(LogNormalAlt2(μr,σr),N)  
        K = rand(LogNormalAlt2(μK,σK),N)  
        dcell = sample(Dcell,N,replace=true,ordered=false)
        for j in 1:N
            p = K[j]*(1.0 - exp(-C*S*r[j]*∫ūdt[idxfine]/K[j]))
            dparticle = sample(Dparticle,Int64(ceil(p)),replace=true,ordered=false)
            #dsim[j] = dcell[j] + sum(dparticle) + (p - floor(p))*dparticle[end]
                if length(dparticle) > 1
                    dsim[j] = dcell[j] + sum(dparticle[1:end-1]) + (p - floor(p))*dparticle[end]
                elseif length(dparticle) == 1
                    dsim[j] = dcell[j] + (p - floor(p))*dparticle[end]
                else
                    dsim[j] = dcell[j] 
                end
        end
        results[idx] = dsim
    end
    return results
end



function hyper_model(θ)
    # assign constant values (Based on Murphy et al., 2024)
    μr = @view θ[1:M]     # mean of individual r per group
    μK = @view θ[M+1:2M]  # mean of individual K per group

    # hyper means and vars
    σr = θ[2M+1] #STD for r within group    
    σK = θ[2M+2] #STD for K within group   
    mr = θ[2M+3]  # mean of r across groups  
    mK = θ[2M+4]  # mean of K across groups  
    sr = θ[2M+5]  # STD of r across groups 
    sK = θ[2M+6]  # STD of K across groups 

    results = Vector{Vector{Vector{Float64}}}(undef, length(T))

    Tfine = 0:60*30:T[end]
    ū = Vector{Vector{Float64}}(undef,length(Tfine)) # integral
    for (idx,t) in enumerate(Tfine)
        reps = Vector{Float64}(undef, M)
        for i in 1:M
            N_prep = 100
            p_prep = Vector{Float64}(undef, N_prep)
            # sample parameter for each individual
            r_prep  = rand(LogNormalAlt2(μr[i],σr),N_prep)  
            K_prep = rand(LogNormalAlt2(μK[i],σK),N_prep)  
            for j in 1:N_prep
                p_prep[j] = solve_ode_model(t,r_prep[j],K_prep[j])
            end
            u = u0 .- p_prep/V
            reps[i] = sum(u)/N_prep
        end
        ū[idx] = reps 
    end
    #TODO: use trap rule to get ∫ū(t)dt
    ∫ūdt = deepcopy(ū)
    for (idx,t) in enumerate(Tfine)
        if idx > 1
            for i in 1:M
                ∫ūdt[idx][i] = ∫ūdt[idx-1][i] + 0.5*(ū[idx][i] + ū[idx-1][i])*(Tfine[idx]-Tfine[idx-1])
            end
        else
            for i in 1:M
                ∫ūdt[idx][i] = 0.5*(ū[idx][i] + u0)*(Tfine[idx])
            end
        end
    end

    for (idx,t) in enumerate(T)
        reps = Vector{Vector{Float64}}(undef, M)
        idxfine = findfirst(Tfine .== T[idx]) # needed for indexing ∫ū
        for i in 1:M
            dsimᵢ = Vector{Float64}(undef, N)
            # sample parameter for each individual
            rᵢ  = rand(LogNormalAlt2(μr[i],σr),N)  
            Kᵢ = rand(LogNormalAlt2(μK[i],σK),N)  
            dcell = sample(Dcell,N,replace=true,ordered=false)
            for j in 1:N
                pᵢ = Kᵢ[j]*(1.0 - exp(-C*S*rᵢ[j]*∫ūdt[idxfine][i]/Kᵢ[j]))
                #println((T[idx],pᵢ,rᵢ[j],Kᵢ[j],S,C,ū[idxfine][i]))
                dparticle = sample(Dparticle,Int64(ceil(pᵢ)),replace=true,ordered=false)
                if length(dparticle) > 1
                    dsimᵢ[j] = dcell[j] + sum(dparticle[1:end-1]) + (pᵢ - floor(pᵢ))*dparticle[end]
                elseif length(dparticle) == 1
                    dsimᵢ[j] = dcell[j] + (pᵢ - floor(pᵢ))*dparticle[end]
                else
                    dsimᵢ[j] = dcell[j] 
                end
            end
            reps[i] = dsimᵢ
        end
        results[idx] = reps 
    end
    return results
end
# ================================================================
# Discrepancy functions
# ================================================================
function makecdf(x::Vector;n=1000)
    n_x = length(x)
    xp = range(minimum(x),maximum(x) + 1e-3,length=n)
    p = [count(x .< xpᵢ) for xpᵢ in xp] / n_x
    itp = LinearInterpolation(xp, p, extrapolation_bc = Interpolations.Flat())
    return itp
end

function addist(cdf::AbstractInterpolation,y::Vector)
    n = length(y)
    cdfs = min.(max.(cdf(sort(y)),1e-9),1-1e-9)
    sqrt(-n - sum( (2(1:n) .- 1.0) / n .* (log.(cdfs) + log.(1.0 .- cdfs[end:-1:1]) )  ))
end

function make_discrepancy(X::Vector{Vector{Float64}}, f=addist; w=[0.5,0.5])
    # Creates discrepancy function Anderson-Darling distance
    Xcdf = [makecdf(X[i]) for i = 1:length(X)]
    function dist(Y::Vector{Vector{Float64}}, _)
        d = 0.0
        for i = 1:length(X)
            d += f(Xcdf[i],Y[i])     
        end
        return d
    end
    return dist
end

function make_discrepancyH(X::Vector{Vector{Vector{Float64}}}, f=addist; w=[0.5,0.5])
    # Creates discrepancy function Anderson-Darling distance
    Xcdf = [[makecdf(X[i][j])  for j=1:length(X[1])] for i = 1:length(X)]
    function dist(Y::Vector{Vector{Vector{Float64}}}, _)
        d = 0.0
        for i = 1:length(X), j = 1:length(X[1])
            d += f(Xcdf[i][j],Y[i][j])     
        end
        return d
    end
    return dist
end


# ================================================================
# SMC parameters
# ================================================================

# SMC set up 
n = 2000 
Rtrial = 10
c = 0.01
pmin = 0.00
a = 0.75
