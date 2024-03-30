using Random # For shuffling
using StatsBase # For sampling
"""
Extraction des données et leurs labels pour les nouvelles datasets (les classes doivent être équilibrées)

"""

function encode_column!(col)
    unique_values = unique(col)
    value_to_int = Dict(value => idx for (idx, value) in enumerate(unique_values))
    return [value_to_int[value] for value in col]
end

function calibrate_split(X, dataset_size)
    # Separate features and labels
    features = X[:, 1:end-1]
    labels = X[:, end]
    
    # Identify all unique labels
    unique_labels = unique(labels)
    
    # Calculate the number of instances to select per label
    num_classes = length(unique_labels)
    instances_per_label = dataset_size ÷ num_classes
    
    # Initialize arrays for the selected indices
    selected_indices = Int[]
    
    # Calibrate instances for each label
    for label in unique_labels
        # Find indices for the current label
        idx_label = findall(x -> x == label, labels)
        
        # Sample instances for the current label
        sampled_indices = sample(idx_label, min(length(idx_label), instances_per_label), replace=false)
        append!(selected_indices, sampled_indices)
    end
    
    # Shuffle the combined indices to mix instances
    Random.shuffle!(selected_indices)
    
    # Create the new calibrated feature matrix and associated labels
    new_X = features[selected_indices, :]
    new_Y = labels[selected_indices]
    
    return new_X, new_Y
end



function read_data(dataSetName,data_size=150)

    if dataSetName=="dry_bean"
        #read data
        file=open("..//data//dry_bean.txt")
        lines=readlines(file)[2:end]
        close(file)

        n=length(lines)
        m=length(split(lines[1],','))

        
        data = Matrix{Any}(undef, n, m)
        for (i, line) in enumerate(lines)
            data[i, :] = split(line, ',')
        end

        #Label encode classes
        data[:, m] = encode_column!(data[:, m])
        #parse data to float
        # Convert each element to Float64
        for i in 1:n
            for j in 1:m-1
                data[i, j] = parse(Float64, data[i, j])
            end
        end


        global X=Matrix{Float64}(zeros(data_size,m-1))
        global Y=Vector{Any}(zeros(data_size))

        X , Y = calibrate_split(data, data_size)
        #println("data: ", X[1:5,:])
        #println("labels: ", Y[1:5])



    
    elseif dataSetName=="mushroom"

        #read data
        file=open("..//data//mushroom.txt")
        lines=readlines(file)[2:end]
        close(file)

        n=length(lines)
        m=length(split(lines[1],','))

        data = Matrix{Any}(undef, n, m)
        for (i, line) in enumerate(lines)
            data[i, :] = split(line, ',')
        end

        # Perform label encoding on each column of the matrix
        for j in 1:m
            data[:, j] = encode_column!(data[:, j])
        end
        # Convert each element to Float64
        for i in 1:n
            for j in 1:m-1
                data[i, j] = float(data[i, j])
            end
        end

        global X=Matrix{Float64}(zeros(data_size,m-1))
        global Y=Vector{Any}(zeros(data_size))

        X , Y = calibrate_split(data, data_size)
        #println("data: ", X[1:5,:])
        #println("labels: ", Y[1:5])



    else
        include("..//data//" * dataSetName * ".txt")
    end

end