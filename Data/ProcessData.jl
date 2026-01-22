using DataFrames, DataFramesMeta, CSV
using FileIO, FCSFiles
using StatsBase

"""
    clip(X;p=[0.001,0.999])
    
Clips the data outside of range specified by `quantile(X,p)`.
    

    clip(X,Y;p=[0.001,0.999])

Clips the data outside of ranges specified by `quantile(X,p)` and `quantile(Y,p)`.

"""
function clip(X;p=[0.001,0.999])
    q = quantile(X,p)
    keep = (X .> q[1]) .& (X .< q[2])
    X[keep]
end
function clip(X,Y;p=[0.001,0.999])
    q1 = quantile(X,p)
    q2 = quantile(Y,p)
    keep = (X .> q1[1]) .& (X .< q1[2]) .& (Y .> q2[1]) .& (Y .< q2[2])
    X[keep],Y[keep]
end



# Create CSV containing all fcs data
files = readdir("Data/FCS")[2:end]
df1   = Array{Any,1}(undef,length(files))   # Store all data
df2   = Array{Any,1}(undef,length(files))   # Store summary (geometric means)
for (i,file) = enumerate(files)

    fcs  = load("Data/FCS/$file")
    df   = DataFrame(fcs.data)
    gain = Dict([fcs.params["\$P$(i)N"] => parse(Float64,fcs.params["\$P$(i)G"]) for i = 1:18]...)
    info = split(file,"_")  # Metadata from filename

    # Normalise BDP and compensatedd Cy5 signal
    signal_BDP = df[:,"488nm (520_40) LinH"] ./ gain["488nm (520_40) LinH"]
    signal_Cy5 = df[:,"FJComp-642nm (676_29) LinH"] ./ gain["642nm (676_29) LinH"]

    # Filter cells with no signal
    keep = (signal_BDP .> 0.0) .& (signal_Cy5 .> 0.0);
    signal_BDP = signal_BDP[keep]
    signal_Cy5 = signal_Cy5[keep]

    # Clip top and bottom 0.1%
    keep = (quantile(signal_BDP,0.001) .< signal_BDP .< quantile(signal_BDP,0.999)) .&  
           (quantile(signal_Cy5,0.001) .< signal_Cy5 .< quantile(signal_Cy5,0.999))
    signal_BDP = signal_BDP[keep]
    signal_Cy5 = signal_Cy5[keep]

    # Convert to data frame (get metadata from filename)
    df1[i] = DataFrame(
        Exp      = fill("DualMarker",                       length(signal_BDP)),
        FCSCond  = fill(info[1],                            length(signal_BDP)),
        Temp     = fill(parse(Float64,info[2][1:end-6]),    length(signal_BDP)),
        CellLine = fill(info[4],                            length(signal_BDP)),
        Quenched = fill(info[3] == "Q",                     length(signal_BDP)),
        Time     = fill(parse(Float64,info[5][1:end-3]),    length(signal_BDP)),
        Repeat   = fill(parse(Int64,info[6][1:end-4]),      length(signal_BDP)),
        Signal_BDP   = signal_BDP,
        Signal_Cy5   = signal_Cy5
    )

    # Summarise data (using MFI/geomean)
    df2[i] = DataFrame(
        Exp      = "DualMarker",
        FCSCond  = info[1], 
        Temp     = parse(Float64,info[2][1:end-6]),
        CellLine = info[4],
        Quenched = info[3] == "Q",
        Time     = parse(Float64,info[5][1:end-3]),
        Repeat   = parse(Int64,info[6][1:end-4]),
        CellCountNZ  = length(signal_Cy5),
        CellCount    = nrow(df),
        Signal_BDP   = geomean(signal_BDP),
        Signal_Cy5   = geomean(signal_Cy5),
    ) 

end

mkdir("Data/CSV")
CSV.write("Data/CSV/DualMarker.csv",vcat(df1...))
CSV.write("Data/CSV/DualMarker_Summary.csv",vcat(df2...))