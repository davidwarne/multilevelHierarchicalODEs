# Multilevel Hierarchical ODEs for Bayesian analysis

This repository contains Julia functions and scripts to implement uncertainty quantiifcation for example flow cytometry data accounting for experimental heterogeneity. 

## Developers

* Xiangrun Zhu [1] (xiangrun.zhu@connect.qut.edu.au),
* Thomas P. Steele [1] (tp.steele@connect.qut.edu.au),
* Alexander P. Browning [2,3] (alex.browning@unimelb.edu.au), https://scholar.google.com/citations?user=Ii-8V2cAAAAJ&hl=en
* David J. Warne [1,3,4] (david.warne@qut.edu.au), https://scholar.google.com.au/citations?user=t8l-kuoAAAAJ&hl=en


1. School of Mathematical Sciences, Faculty of Science, Queensland Univeristy of Technology, Australia
2. School of Mathematics and Statistics, University of Melbourne, Australia
3. ARC Centre of Excellence for the Mathematical Analysis of Cellular Systems
4. Centre for Data Science, Queensland University of Technology, Australia

## Citation Information

This code is provided as supplementary information to the paper,

David J. Warne, Xiangrun Zhu, Thomas P. Steele, Stuart T. Johnston, Scott A. Sisson, Matthew Faria, Ryan J. Murphy, and Alexander P. Browning. A multilevel hierarchical framework for quantification of experimental heterogeneity. bioRxiv preprint (https://doi.org/10.64898/2025.12.21.695338) 

## Licensing
This source code is licensed under the GNU General Public License Version 3.
Copyright (C) 2026 Thomas P. Steele and David J. Warne

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

## Contents

```bash
The directory structure is as follows
|-- startup.jl                              Adds local modules functions to the Julia Path
|-- Plotting/                               Contains code to produce figures 3 to 10 of the paper
|-- Experiments_synthetic_internalisation/  Code for generating synthetic data and running inference experiments for the internalisation model (note: requires HPC)
|-- Experiments_synthetic_affinity/         Code for generating synthetic data and running inference experiments for the pariticle-cell interaction model (note: requires HPC)
|-- results_final_intern/                   Inference results *.jld2 files for for plotting internalisation results.
|-- results_final_affinity/                 Inference results *.jld2 files for for plotting particle-cell interaction results.
|-- Data/                                   Code to extract data for experiments 
|-- Example_real_data/                      Example analysis code for real internalisation data
|-- Modules/                                Implementation modules for all methods in paper
    |-- Dist/
    |-- SMC/
```
## Usage

1. Browse to the repository folder multilevelHierarchicalODEs/
2. Start Julia
3. In the Julia REPL, in package mode (']') run 
   `pkg> activate .`
   `pkg> instantiate`
   to set up the Julia environment.
4. Back in Julia mode, run
   `julia> include("./startup.jl")'
   to set up the paths of local modules.
5. Reproduce a figure e.g.,  
   `julia> include("./Plotting/plot_maringals_and_pps_particle_affinity_results.jl")` 
   generates Figs 3 to 6 in the paper.


