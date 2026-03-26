#=
# Create functions 
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

log_q = (θp,θc,Σ) -> logpdf(MultivariateNormal(θc,Σ),θp)
q = (θc,Σ) -> rand(MultivariateNormal(θc,Σ))

prior_dist = product_distribution(
    Uniform(0.0   , 0.3    ),  # 1.  σλ
    Uniform(0.0   , 0.1    ),  # 2.  σβ
    Uniform(0.0   , 1.0    ),  # 3.  σR
    Uniform(0.0   , 0.5    ),  # 4.  μλ
    Uniform(0.0   , 0.2    ),  # 5.  μβ
    Uniform(0.0   , 1.0    ),  # 6.  μR
    Uniform(0.0   , 10.0   ),  # 7.  σ₁
    Uniform(0.0   , 3.0    ),  # 8. σ₂
    Uniform(5000.0, 12000.0),  # 9. α₁
    Uniform(20.0  , 80.0   ),  # 10. α₂
    Uniform(0.0   , 0.15   ),   # 11. p
    Uniform(-2.0  , 1.0    ),  # 12.  ωλ
    Uniform(-2.0  , 1.0    )  # 13.  ωβ
)

function sample_prior()
    return rand(prior_dist)
end

function log_prior(θ)
    # extract hyper parameters
    σλ = θ[1]    # STD of individual λ
    σβ = θ[2]    # STD of individual β
    σR = θ[3]    # STD of individual R
    σ₁ = θ[7]    # STD of individual observations
    σ₂ = θ[8]    # STD of individual observations
    
    # Ensure standard deviations are greater than 0
    if σλ < 0 || σβ < 0 || σR < 0 || σ₁ < 0 || σ₂ < 0
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
    μλ = @view θ[1:M]     # mean of individual λ per group
    μβ = @view θ[M+1:2M]  # mean of individual β per group
    μR = @view θ[2M+1:3M]  # mean of individual β per group

    # hyper means and vars
    σλ = θ[3M+1]    # STD of individual λ within groups
    σβ = θ[3M+2]    # STD of individual β within groups
    σR = θ[3M+3]
    mλ = θ[3M+4]    # mean of λ means
    mβ = θ[3M+5]    # mean of β means
    mR = θ[3M+6]
    sλ = θ[3M+7]    # STD of β means
    sβ = θ[3M+8]    # STD of λ means 
    sR = θ[3M+9]

    # Equation parameters
    σ₁ = θ[3M+10]   # STD of individual observations
    σ₂ = θ[3M+11]   # STD of individual observations
    α₁ = θ[3M+12]   # contant of proportionality
    α₂ = θ[3M+13]   # contant of proportionality
    p  = θ[3M+14]   # proportion recycled
    ωλ  = θ[3M+15]   # skewness of λ within groups
    ωβ  = θ[3M+16]   # skewness of β within groups
    
    if σλ < 0 || σβ < 0 ||σR < 0 || sλ < 0 || sβ < 0 || sR < 0 || σ₁ < 0 || σ₂ < 0 || any(μλ .< 0) || any(μβ .< 0)
        log_p = -Inf
    else
        log_p = log_hyperprior([σλ,σβ,σR,mλ,mβ,mR,sλ,sβ,sR,σ₁,σ₂,α₁,α₂,p,ωλ,ωβ]) + sum(logpdf.(Normal(mλ, sλ), μλ)) + sum(logpdf.(Normal(mβ, sβ), μβ)) + sum(logpdf.(Normal(mR,sR),μR))
    end
    return log_p
end

function sample_hyper_prior()
    θ = zeros(3M+16)
    λ = rand(hyperprior_dist);
    θ[3M+1:3M+16] = λ
    mλ = θ[3M+4]    # mean of λ means
    mβ = θ[3M+5]    # mean of β means
    mR = θ[3M+6]
    sλ = θ[3M+7]    # STD of β means
    sβ = θ[3M+8]    # STD of λ means 
    sR = θ[3M+9]

    θ[1:M]     = rand(TruncatedNormal(mλ,sλ,0,Inf),M)
    θ[M+1:2M]  = rand(TruncatedNormal(mβ,sβ,0,Inf),M)
    θ[2M+1:3M] = rand(TruncatedNormal(mR,sR,0,Inf),M)
    return θ
end

# ================================================================
# Model functions
# ================================================================

function solve_ode_model(t::Float64, λ::Float64, β::Float64, p::Float64)
    I = (λ*(λ-β+β*p)*(t*β^3*p^2+t*β^2*λ*p^2+t*β^2*λ*p+t*β*λ^2*p-β*λ*p^2+β*λ-λ^2*p+λ^2)+β^2*λ^2*(1-p)*exp(-t*(λ+β*p))+(2*β*λ^3*p^2-2*β*λ^3*p-β^2*λ^2*p^2+β^2*λ^2*p^3+λ^4*p-λ^4)*exp(-β*t)) / ((β+λ)*(λ+β*p)^2*(λ-β +β*p))
    S = β*((p*(β+λ)*(λ-β+β*p)+λ*β*(λ*p-1)*exp(-λ*t-β*p*t)+λ*p*exp(-β*t))/((β+λ)*(λ+β*p)*(λ-β+β*p)))
    return [S,I]
end

# simulate individual models
function model(θ)
    # forward sim in time
    # extract hyper parameters

    σλ = θ[1]   # STD of λ value for cells
    σβ = θ[2]   # STD of β value for cells
    σR = θ[3]   # STD of R value for cells

    # hyper means and vars
    μλ = θ[4]   # Mean of λ for cells
    μβ = θ[5]   # Mean of β for cells
    μR = θ[6]   # Mean of R for cells

    # equation parameters
    σ₁ = θ[7]   # Error of individual observations
    σ₂ = θ[8]   # Error of individual observations
    α₁ = θ[9]   # Contant of proportionality
    α₂ = θ[10]  # Contant of proportionality
    p  = θ[11]  # Proportion recycled
    ωλ = θ[12]  # skewness of λ value for cells
    ωβ = θ[13]  # skewness of β value for cells
    results = Vector{Vector{Vector{Vector{Float64}}}}(undef, length(T))

    # Sample 
    Sᵢ = Vector{Float64}(undef, 2N)
    Iᵢ = Vector{Float64}(undef, 2N)

    for (idx, t) in enumerate(T)
        # sample parameter for each individual
        λᵢ  = rand(GammaAlt(μλ,σλ,ωλ),2N)  # Fix T normal + gamma
        βᵢ  = rand(GammaAlt(μβ,σβ,ωβ),2N)  # Truncated normal 
        Rᵢ  = rand(LogNormalAlt(1.0,μR,σR),2N) 

        κ₁  = rand(Normal(0,σ₁),2N)
        κ₂  = rand(Normal(0,σ₂),2N)

        AF₁ = sample(1:length(E[1]),N)
        AF₂ = sample(1:length(E[2]),N)

        # Sample 
        for j in 1:2N
            Sᵢ[j], Iᵢ[j] = solve_ode_model(t, λᵢ[j], βᵢ[j], p)
        end
        # Total antibody
        V₁ = max.(0, (Sᵢ[1:N] .+ Iᵢ[1:N]) .* Rᵢ[1:N])
        V₂ = max.(0, (Sᵢ[N+1:end] .+ Iᵢ[N+1:end]) .* Rᵢ[N+1:end])
        # Total internal antibody
        V̄₂ = max.(0, Rᵢ[N+1:end] .* (Iᵢ[N+1:end] .+ (1-η) * Sᵢ[N+1:end]))

        # Q ~ Unquenched FIP-Cy5 
        Q = α₁ * V₁ + E[1][AF₁] + κ₁[1:N]
        # Q̄ ~ Quenched FIP-Cy5 
        Q̄  = α₁ * V̄₂ + E[1][AF₂] + κ₁[N+1:end]

        # U ~ Unquenched BODIPY FL
        U = α₂ * V₁ + E[2][AF₁] + κ₂[1:N]
        # Ū ~ Quenched BODIPY FL
        Ū = α₂ * V₂ + E[2][AF₂] + κ₂[N+1:end]

        results[idx] = [[Q̄, Ū], [Q, U]]
    end
    return results
end

function hyper_model(θ)
    μλ = @view θ[1:M]     # mean of individual λ per group
    μβ = @view θ[M+1:2M]  # mean of individual β per group
    μR = @view θ[2M+1:3M]  # mean of individual R per group

    # hyper means and vars
    σλ = θ[3M+1]    # STD of individual λ within groups
    σβ = θ[3M+2]    # STD of individual β within groups
    σR = θ[3M+3]    # STD of individual β within groups

    # equation parameters
    σ₁ = θ[3M+10]   # STD of individual observations
    σ₂ = θ[3M+11]   # STD of individual observations
    α₁ = θ[3M+12]   # contant of proportionality
    α₂ = θ[3M+13]   # contant of proportionality
    p  = θ[3M+14]   # proportion recycled
    ωλ = θ[3M+15]  # skewness of λ value for cells
    ωβ = θ[3M+16]  # skewness of β value for cells

    results = Vector{Vector{Vector{Vector{Vector{Float64}}}}}(undef, length(T))

    for (idx,t) in enumerate(T)
        reps = Vector{Vector{Vector{Vector{Float64}}}}(undef, M)
        for i in 1:M
            Q = Vector{Float64}(undef, N)
            Q̄ = Vector{Float64}(undef, N)
            U = Vector{Float64}(undef, N)
            Ū = Vector{Float64}(undef, N)

            # Sample 
            Sᵢ = Vector{Float64}(undef, 2N)
            Iᵢ = Vector{Float64}(undef, 2N)
            
            # sample parameter for each individual
            λᵢ  = rand(GammaAlt(μλ[i],σλ,ωλ),2N)  
            βᵢ  = rand(GammaAlt(μβ[i],σβ,ωβ),2N)  
            Rᵢ  = rand(LogNormalAlt(1.0,μR[i],σR),2N)  

            κ₁  = rand(Normal(0,σ₁),2N)
            κ₂  = rand(Normal(0,σ₂),2N)

            AF₁ = sample(1:length(E[1]),N)
            AF₂ = sample(1:length(E[2]),N)

            for j in 1:2N
                Sᵢ[j], Iᵢ[j] = solve_ode_model(t, λᵢ[j], βᵢ[j], p)
            end
            # Total antibody
            V₁ = max.(0, (Sᵢ[1:N] .+ Iᵢ[1:N]) .* Rᵢ[1:N])
            V₂ = max.(0, (Sᵢ[N+1:end] .+ Iᵢ[N+1:end]) .* Rᵢ[N+1:end])
            # Total internal antibody
            V̄₂ = max.(0, Rᵢ[N+1:end] .* (Iᵢ[N+1:end] .+ (1-η) * Sᵢ[N+1:end]))

            # Q ~ Unquenched FIP-Cy5 
            Q[1:N] = α₁ * V₁ + E[1][AF₁] + κ₁[1:N]
            # Q̄ ~ Quenched FIP-Cy5 
            Q̄[1:N]  = α₁ * V̄₂ + E[1][AF₂] + κ₁[N+1:end]

            # U ~ Unquenched BODIPY FL
            U[1:N] = α₂ * V₁ + E[2][AF₁] + κ₂[1:N]
            # Ū ~ Quenched BODIPY FL
            Ū[1:N] = α₂ * V₂ + E[2][AF₂] + κ₂[N+1:end]

            reps[i] = [[Q̄, Ū], [Q, U]]
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

function make_discrepancy(X::Vector{Vector{Vector{Vector{Float64}}}}, f=addist; w=[0.5,0.5])
    # Creates discrepancy function Anderson-Darling distance
    Xcdf = [[makecdf.(X[i][j]) for j = 1:length(X[1])] for i = 1:length(X)]
    Xcor = [[cor(X[i][j]...) for j = 1:length(X[1])] for i = 1:length(X)]
    function dist(Y::Vector{Vector{Vector{Vector{Float64}}}}, _)
        d = 0.0
        for i = 1:length(X), j = 1:length(X[1])
            d += 2 * w[1] * f(Xcdf[i][j][1],Y[i][j][1]) +    # Quenchable
            1 * w[1] * f(Xcdf[i][j][2],Y[i][j][2]) +    # Unquenchable
            1 * w[2] * norm(Xcor[i][j] - cor(Y[i][j]...))  # Correlations
        end
        return d
    end
    return dist
end

function make_discrepancyH(X::Vector{Vector{Vector{Vector{Vector{Float64}}}}}, f=addist; w=[0.5,0.5])
    # Creates discrepancy function Anderson-Darling distance
    Xcdf = [[[makecdf.(X[i][j][k]) for k = 1:length(X[1][1])] for j = 1:length(X[1])] for i = 1:length(X)]
    Xcor = [[[cor(X[i][j][k]...) for k = 1:length(X[1][1])] for j = 1:length(X[1])] for i = 1:length(X)]
    function dist(Y::Vector{Vector{Vector{Vector{Vector{Float64}}}}}, _)
        d = 0.0
        for i = 1:length(X), j = 1:length(X[1]), k = 1:length(X[1][1])
            d += 2 * w[1] * f(Xcdf[i][j][k][1],Y[i][j][k][1]) +    # Quenchable
            1 * w[1] * f(Xcdf[i][j][k][2],Y[i][j][k][2]) +    # Unquenchable
            1 * w[2] * norm(Xcor[i][j][k] - cor(Y[i][j][k]...))  # Correlations
        end
        return d
    end
    return dist
end
#
# ================================================================
# SMC parameters
# ================================================================

# SMC set up 
n = 2000 
Rtrial = 10
c = 0.01
pmin = 0.001
a = 0.75
