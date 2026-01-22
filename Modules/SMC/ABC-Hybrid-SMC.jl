#=
# Functions for hybrid Approximate Bayesian Computation/Likelihood-base 
# Sequential Monte Carlo Sampler.
#
# author: David J. Warne (david.warne@qut.edu.au)
#                         School of Mathematical Sciences
#                         Science and Engineering Faculty
#                         Queensland University of Technology
=#

@doc """
    ABCHybridSMC(log_p::Function,p::Function,log_q::Function,q::Function,s::Function,
                      ρ::Function,D::Array{Float64},n::Int,Rtrial::Int,
                     c::Float64,pmin::Float64,a::Float64)
Generates `T` iterations of the ABC/Likelihood based SMC for a hierarchical model 
p(ϕⁱ,λ | ϵ ≥ ∑ⱼ ρ(Dsʲ,Dʲ)) where Dʲ is observed data, Dsʲ ∼ s(Dʲ | θʲ) and i
p(θʲ|ϕⁱ)p(ϕⁱ|λ) is the forward model, and ρ(⋅,D) is an appropirately chosen 
discrepency metric to ensure that p(ϕⁱ,λ | ϵ ≥ ∑ⱼ ρ(Dsʲ,Dʲ)) → p(ϕⁱ,λ | Dʲ) as ϵ → 0. 

Inputs:\n
    `log_p` - the log prior density function for ϕ¹,...,ϕᴹ, and hyper parameters λ
              e.g., p(ϕ¹|λ) p(ϕ²|λ) ⋯ p(ϕᴹ|λ)p(λ) 
    `p` - prior sampler 
    `log_q` - the log proposal density function in ϕ¹,...,ϕᴹ, λ
    `q` - proposal sampler 
    `s` - forwards problem simulation (i.e., generates simulated Data given ϕ¹,...,ϕᴹ, λ)
    `ρ` - the descrepancy metric
    `ϵ` - acceptance threshold
    `D` - observed data
    `n` - number of particles for SMC 
    `Rtrial` - tuning steps for coosing R
    `c` - tuning parameter for choosing R
    `pmin` - minimum acceptance probability
    `a` - splitting percentage per iteration
    callBackName - Save JLD2 file of results with this name 
    `maximumTime` - optional break condition on maximum time
    `ϵTarget` - optional break condition on target ϵ

Outputs:\n
    `θt` - array of samples θt[:,i] is the state at the i-th 
            iteration. θt[j,:] is the j-th marginal of θt.
"""
function ABCHybridSMC(log_p::Function,p::Function,log_q::Function,q::Function,s::Function,
                      ρ::Function,D::Any,n::Int,Rtrial::Int,c::Float64,
                     pmin::Float64,a::Float64, callBackName::String; maximumTime = Inf, ϵTarget = -Inf)
    ϵt = 0
    pacc = Inf
    # n to accept and leave each iterations
    na = Int(floor(a*n))
    nl = n - na
    # initial rejection sampling 
    m, = size(p())
    θt = zeros(Float64,m,n)
    ρₙ= zeros(Float64,n)
    print("Initialising particles...")
    @threads for i in 1:n
    #for i in 1:n
        θt[:,i] = p() # here θ = ϕ¹,...,ϕᴹ, λ
        Ds = s(θt[:,i])
        ρₙ[i]= ρ(Ds,D)
       # println(ρₙ[i])
    end
    println("done!")
    
    iter = 0
    println("Starting sequential sampling...")
    ctime = 0
    while pacc >= pmin
        #sort particles by discrepancy
        I = sortperm(ρₙ)
        ρₙ= ρₙ[I]
        θt = θt[:,I] 
        ϵt = ρₙ[nl]
        #resample rejected particles from accepted with replacement
        J = sample(1:nl,na,replace=true)
        θt[:,(nl+1):n] = θt[:,J]
        #adapt proposal covariance
        Σ = ((2.38^2)/m)*cov(θt,dims=2)
        iter+=1
        println("--> Iteration ",iter)
        println("    Current threshold ϵt = ", ϵt)

        time1 = @elapsed begin
            # determine length of R
            pacc = zeros(na)
            @threads for j in nl+1:n
            #for j in nl+1:n
                for _ in 1:Rtrial
                    #generate proposals
                    θprop = q(θt[:,j],Σ)
                    u = rand(Uniform(0,1))
                    if log(u) <= min(0,(log_q(θt[:,j],θprop,Σ) + log_p(θprop))-(log_q(θprop,θt[:,j],Σ)+log_p(θt[:,j])))
                        Ds = s(θprop)
                        ρprop = ρ(Ds,D)
                        #println([θt[:,j],θprop])
                        #println([ρprop,ϵt])
                        if ρprop <= ϵt
                            θt[:,j] = θprop
                            ρₙ[j] = ρprop
                            pacc[j-nl] += 1
                        end
                    end
                end
            end
            pacc = sum(pacc)/(Rtrial*na)
        end
        println("    Trial acceptance probability pacc = ",pacc)
        println("    Trial time = ",time1)
        # set R based on propability of a particle not moving being < c
        R = ceil(log(c)/(log(1-pacc)))
        println("    Iterations remaining R = ",R-Rtrial)
        println("    Expected time remaining = ", (R-Rtrial)*na*(time1/(na*Rtrial)))
        # complete the move step
        pacc = 0
        pacc = zeros(na)
        time2 = @elapsed begin
            @threads for j in nl+1:n
                for _ in 1:(R - Rtrial)
                    #generate proposals
                    θprop = q(θt[:,j],Σ)
                    u = rand(Uniform(0,1))
                    if log(u) <= min(0,(log_q(θt[:,j],θprop,Σ) + log_p(θprop))-(log_q(θprop,θt[:,j],Σ)+log_p(θt[:,j])))
                        Ds = s(θprop)
                        ρprop = ρ(Ds,D)
                        if ρprop <= ϵt
                            θt[:,j] = θprop
                            ρₙ[j] = ρprop
                            pacc[j-nl] += 1
                        end
                    end
                end
            end
            pacc = sum(pacc)/((R-Rtrial)*na)
        end
        time = time1 + time2
        ctime += time
        println("    Current threshold ϵt = ", ϵt)
        println("    Acceptance probability pacc = ",pacc)
        println("    Iteration time = ",time)
        println("<--")
        @save "$(callBackName)_err_$(round(Int,ϵt))_iter_$(iter).jld2" θt ρₙ # epsilon in file name + iter in file name

        if ctime > maximumTime * 60 || ϵt < ϵTarget
            break
        end
    end
    println("Sampling done!")
    println("Total time = ",ctime)
    return θt, ρₙ, ϵt
end
