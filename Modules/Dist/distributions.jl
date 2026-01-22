#=

    distributions.jl

    Contains code to create alternatively parameterised distributions, including copulae.

    Author:     Alexander P. Browning
                ======================
                School of Mathematical Sciences
                Queensland University of Technology
                ======================
                ap.browning@icloud.com
                alexbrowning.me

=#

import Base: rand, minimum, maximum
import Distributions: pdf, cdf
import Statistics: mean, var, std, quantile
import StatsBase: skewness, sample

##############################################################
## Alternative parameterisation of the gamma distribution
##############################################################
struct GammaAlt{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    ω::T
    d::ContinuousUnivariateDistribution  # Underlying (truncated) Gamma distribution
    GammaAlt{T}(μ,σ,ω,d) where {T} = new{T}(μ,σ,ω,d) 
end
struct GammaAltNegative{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    ω::T
    d::ContinuousUnivariateDistribution  # Underlying (truncated) Gamma distribution
    GammaAltNegative{T}(μ,σ,ω,d) where {T} = new{T}(μ,σ,ω,d) 
end

"""
    GammaAlt(μ,σ,ω)

Construct a truncated (x > 0) Gamma distribution `d` where the mean, standard deviation
and skewness of the untruncated distribution are given by μ, σ and ω.

"""
function GammaAlt(μ::T,σ::T,ω::T) where {T <: Real}
    μ > 0.0 || error("Mean must be positive.")
    σ > 0.0 || error("Standard deviation must be positive.")
    if ω < 0
        α = 4/ω^2
        θ = -σ * ω / 2
        d = Truncated(Gamma(α, θ) - α*θ - μ,-Inf,0.0)
        GammaAltNegative{T}(μ,σ,ω,d)
    else
        α = 4/ω^2
        θ = σ * ω / 2
        d = Truncated(Gamma(α, θ) - α*θ + μ,0.0,Inf)
        GammaAlt{T}(μ,σ,ω,d)
    end
end

#### Evaluation
rand(rng::AbstractRNG, d::GammaAltNegative) = -rand(rng,d.d)
pdf(d::GammaAltNegative,x::Real) = pdf(d.d,-x)
logpdf(d::GammaAltNegative,x::Real) = logpdf(d.d,-x)
cdf(d::GammaAltNegative,x::Real) = 1 .- cdf(d.d,-x)
quantile(d::GammaAltNegative,p::AbstractArray) = -quantile(d.d,1.0 .- p)
quantile(d::GammaAltNegative,p::Number) = -quantile(d.d,1.0 .- p)
minimum(d::GammaAltNegative) = 0.0
maximum(d::GammaAltNegative) = Inf

rand(rng::AbstractRNG, d::GammaAlt) = rand(rng,d.d)
pdf(d::GammaAlt,x::Real) = pdf(d.d,x)
logpdf(d::GammaAlt,x::Real) = logpdf(d.d,x)
cdf(d::GammaAlt,x::Real) = cdf(d.d,x)
quantile(d::GammaAlt,p::AbstractArray) = quantile(d.d,p)
quantile(d::GammaAlt,p::Number) = quantile(d.d,p)
minimum(d::GammaAlt) = 0.0
maximum(d::GammaAlt) = Inf


##############################################################
## Alternative parameterisation of the Log Normal distribution
##############################################################
"""
    LogNormalAlt(μ,μ₁,σ₁)

Construct a truncated (x > 0) Log-Normal distribution `d` where the mean of the 
untruncated distribution is `μ`. `μ₁` and `σ₁` are the standard Log Normal parameters
in the unshifted distribution.
"""
LogNormalAlt(μ,μ₁,σ₁) = Truncated(LogNormal(μ₁,σ₁) - exp(μ₁ + σ₁^2/2) + μ,0,Inf)

"""
    LogNormalAlt2(μX,σX)

Construct a Log-Normal distribution `d` where the mean of the 
 distribution is `μX` and `σX` is the standard deviation of the  Log Normal 
 random Variable X.
"""
LogNormalAlt2(μX,σX) = LogNormal(log((μX^2)/sqrt(μX^2 + σX^2)),sqrt(log(1 + (σX/μX)^2)))

Distributions.cdf(x::Vector) = invperm(sortperm(x)) / length(x) .- 1 / 2length(x)

function halfCauchy(μ,σ)
    return Truncated(Cauchy(μ,σ), μ, Inf)
end

function TruncatedNormal(μ,σ,l,u)
    return Truncated(Normal(μ,σ), l, u)
end
