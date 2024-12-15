include("building_tree_.jl")
include("utilities.jl")
include("merge.jl")
include("read_data.jl")
using Distances, StatsBase

function main_merge_sC(inegalite=false,save=true)
    for dataSetName in ["iris", "seeds", "wine","dry_bean","mushroom"]
        #result_path = "..//results"
        #file_name= "results_main_merge_spectral_clustering-"*dataSetName*".txt"
        #res=open(joinpath(result_path,file_name),"w")
        #print(res,"=== Dataset ", dataSetName)
        if inegalite &&save
            res=open("..//results-mergesC-inegalites-"*dataSetName*".txt","w")
            res_save=open("..//save-results-mergesC-inegalites-"*dataSetName*".txt","w")
        elseif inegalite && !(save)
            res=open("..//results-mergesC-inegalites-"*dataSetName*".txt","w")
        elseif save && !inegalite 
            res_save=open("..//save-results-mergesC-"*dataSetName*".txt","w")
            res=open("..//results-mergesC-"*dataSetName*".txt","w")
        else 
            res=open("..//results-mergesC-"*dataSetName*".txt","w")
        end
        print("=== Dataset ", dataSetName)
        println(res,"=== Dataset ", dataSetName)  

        # Préparation des données
        read_data(dataSetName,150)
        # Ramener chaque caractéristique sur [0, 1]
        reducedX = Matrix{Float64}(X)
        for j in 1:size(X, 2)
            reducedX[:, j] .-= minimum(X[:, j])
            reducedX[:, j] ./= maximum(X[:, j])
        end

        train, test = train_test_indexes(length(Y))
        X_train = reducedX[train, :]
        Y_train = Vector{Any}(Y[train])
        X_test = reducedX[test, :]
        Y_test = Vector{Any}(Y[test])
        classes = Vector{Any}(unique(Y))

        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")
        println(res," (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")

        # Temps limite de la méthode de résolution en secondes        
        time_limit = 900

        #save table
        if save 
            println(res_save,"\\begin{tabular}{cc||ccccc|ccccc}")
            println(res_save,"\\hline")
            if inegalite 
                test="Activated"
            else
                test="Deactivated"
            end
            println(res_save,"\\multicolumn{2}{|c|}{Dataset=",dataSetName,"}&\\multicolumn{2}{|c|}{Train size=",size(X_train, 1),"}&\\multicolumn{2}{|c|}{Test size=",size(X_test, 1),"}&\\multicolumn{2}{|c|}{Feature count =", size(X_train, 2),"}&\\multicolumn{2}{|c|}{Time limit =",time_limit, "}&\\multicolumn{2}{|c|}{Inequalities =",test, "}\\\\ \\hline\\hline")
        end

        for k in [1, 5, 10, 15]
            if save 
                println(res_save,"k&$k&\\multicolumn{5}{c}{Univarié}&\\multicolumn{5}{c}{Multivarié}\\\\\\hline")
                println(res_save,"Sigma&Nb Clusters&Solve time&Gap(\\%)&Train/test&Node count&Nb constraints&Solve time&Gap(\\%)&Train/test&Node count& nb constraints\\\\")
            end
            println("\tk = ", k)
            println(res,"\tk = ", k)
            println("\t\tUnivarié")
            println(res,"\t\tUnivarié")
            results_uni=testMergesC(X_train, Y_train, X_test, Y_test, k, classes,res;time_limit = time_limit, isMultivariate = false,inegalite)
            println("\t\tMultivarié")
            println(res,"\t\tMultivarié")
            results_multi=testMergesC(X_train, Y_train, X_test, Y_test, k, classes,res; time_limit = time_limit, isMultivariate = true,inegalite)
            if save 
                for i in eachindex(results_uni)
                    println(res_save,results_uni[i],results_multi[i])
                end
                if k!=15
                    println(res_save,"\\hline")
                end
            end
                
        end
        if save 
            println(res_save, "\\end{tabular}")
            close(res_save)
        end
        close(res)
    end
end 

function testMergesC(X_train, Y_train, X_test, Y_test, k, classes,res; time_limit::Int=-1, isMultivariate::Bool = false,inegalite::Bool = false)

    
    println("\t\t\tsigma\t\t\t# clusters\tGap")
    println(res,"\t\t\tsigma\t\t# clusters\tGap")
    distances = pairwise(Euclidean(), X_train, X_train, dims=1)

    
    
    # Calculate the median distance between neighboring data points
    median_distance = median(distances[triu(ones(size(distances)), 1) .!= 0])

    # Define the lower and upper bounds of the search range
    sigma_min = median_distance / 10
    sigma_max = 10 * median_distance

    results=["" for _ in sigma_min:(sigma_max-sigma_min)/10:sigma_max]
    index=1
    
   
    for sigma in sigma_min:(sigma_max-sigma_min)/10:sigma_max
        print("\t\t\t ", round(sigma, digits= 3),"\t\t\t")
        print(res,"\t\t\t ", round(sigma,digits=3), "%\t\t")
          
        clusters = spectralClustering(X_train, Y_train, k,sigma)
    
        print(length(clusters), " \t\t")
        print(res,string(length(clusters))*" clusters\t")
        T, obj, resolution_time, gap,nb,nb_cons  = build_tree_cluster(clusters, 4,inegalite, classes, multivariate = isMultivariate, time_limit = time_limit)
        print(round(gap, digits = 3), "%\t") 
        print(res,round(gap, digits = 3), "%\t")
        print("Erreurs train/test : ", prediction_errors(T,X_train,Y_train, classes))
        print(res,"Erreurs train/test : ", prediction_errors(T,X_train,Y_train, classes))
        print("/", prediction_errors(T,X_test,Y_test, classes), "\t")
        print(res,"/", prediction_errors(T,X_test,Y_test, classes), "\t")
        #print(" accuracy ",round((size(X_test,1)-prediction_errors(T,X_test,Y_test, classes))/size(X_test,1), digits=3),"\t\t")
        #print(res," accuracy ",round((size(X_test,1)-prediction_errors(T,X_test,Y_test, classes))/size(X_test,1), digits=3),"\t\t")
        println("\t", round(resolution_time, digits=1), "s")
        println(res,"\t",round(resolution_time, digits=1), "s")
        if !isMultivariate
            results[index]=string(sigma)*"&"*string(length(clusters))*"&"*string(round(resolution_time, digits=1))*"&"*string(round(gap, digits = 1))*"&"*string(prediction_errors(T,X_train,Y_train, classes))*"/"*string(prediction_errors(T,X_test,Y_test, classes))*"&"*string(nb)*"&"*string(nb_cons)*"&"
        else 
            results[index]=string(round(resolution_time, digits=1))*"&"*string(round(gap, digits = 1))*"&"*string(prediction_errors(T,X_train,Y_train, classes))*"/"*string(prediction_errors(T,X_test,Y_test, classes))*"&"*string(nb)*"&"*string(nb_cons)*"\\\\"
        end
        index+=1
    end
    println() 
    println(res)
    return results
end   


   