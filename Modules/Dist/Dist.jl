

module Dist

using Base
using Distributions
using Statistics
using StatsBase
using Random

export GammaAlt, GammaAltNegative, LogNormalAlt, LogNormalAlt2, halfCauchy, TruncatedNormal

include("distributions.jl")
    
end
