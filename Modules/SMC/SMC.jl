"""
A Julia module to demonstrate various MCMC schemes, both exact and approximate
sampling is supported.

Supported Samplers:
    Metropolis-Hastings MCMC, Approximate Bayesian Computation MCMC, 
    Pseudo-Marginal (Particle) MCMC 
"""
module SMC
#=
# author: David J. Warne (david.warne@qut.edu.au)
#                         School of Mathematical Sciences
#                         Science and Engineering Faculty
#                         Queensland University of Technology
=#
using StatsBase
using Distributions
using Base.Threads
export ABCHybridSMC
using JLD2

include("ABC-Hybrid-SMC.jl")

end
