include("building_tree_.jl")
include("utilities.jl")
include("merge.jl")
include("read_data.jl")

function main_merge(inegalite=false,save=true)
    for dataSetName in ["iris","seeds","wine"]
    #for dataSetName in ["iris"]
        if inegalite &&save
            res=open("..//results-merge-inegalites-"*dataSetName*".txt","w")
            res_save=open("..//save-results-merge-inegalites-"*dataSetName*".txt","w")
        elseif inegalite && !(save)
            res=open("..//results-merge-inegalites-"*dataSetName*".txt","w")
        elseif save && !inegalite 
            res_save=open("..//save-results-merge-"*dataSetName*".txt","w")
            res=open("..//results-merge-"*dataSetName*".txt","w")
        else 
            res=open("..//results-merge-"*dataSetName*".txt","w")
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
        X_train = reducedX[train,:]
        Y_train = Vector{Any}(Y[train])
        X_test = reducedX[test,:]
        Y_test = Vector{Any}(Y[test])
        classes = Vector{Any}(unique(Y))
        
        # Temps limite de la méthode de résolution en secondes        
        time_limit = 900
        if inegalite 
            ch=", Avec inégalités valides"
        else 
            ch=""
        end
        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2),", time limit: ", time_limit,ch, ")")
        println(res," (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2),", time limit: ", time_limit, ch,")")
        
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
        for D in 2:4
            if save 
                println(res_save,"D&$D&\\multicolumn{5}{c}{Univarié}&\\multicolumn{5}{c}{Multivarié}\\\\\\hline")
                println(res_save,"Gamma&Nb Clusters&Solve time&Gap(\\%)&Train/test&Node count&Nb constraints&Solve time&Gap(\\%)&Train/test&Node count& nb constraints\\\\")
            end
            println("\tD = ", D)
            println(res,"\tD = ", D)
            println("\t\tUnivarié")
            println(res,"\t\tUnivarié")
            results_uni=testMerge(X_train, Y_train, X_test, Y_test, D, classes,res;time_limit = time_limit, isMultivariate = false,inegalite)
            println("\t\tMultivarié")
            println(res,"\t\tMultivarié")
            results_multi=testMerge(X_train, Y_train, X_test, Y_test, D, classes,res; time_limit = time_limit, isMultivariate = true,inegalite)
            if save 
                for i in eachindex(results_uni)
                    println(res_save,results_uni[i],results_multi[i])
                end
                if D!=4
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

function testMerge(X_train, Y_train, X_test, Y_test, D, classes,res; time_limit::Int=-1, isMultivariate::Bool = false,inegalite::Bool = false)
    # Pour tout pourcentage de regroupement considéré
    println("\t\t\tGamma\t\t# clusters\tGap")
    println(res,"\t\t\tGamma\t\t# clusters\tGap")
    results=["" for _ in 0:0.2:1]
    index=1
    for gamma in 0:0.2:1
        print("\t\t\t", gamma * 100, "%\t\t")
        print(res,"\t\t\t", gamma * 100, "%\t\t")
        clusters = simpleMerge(X_train, Y_train, gamma)
        print(length(clusters), " clusters\t")
        print(res,string(length(clusters))*" clusters\t")
        T, obj, resolution_time, gap,nb,nb_cons = build_tree_cluster(clusters,D,inegalite,classes; multivariate=isMultivariate, time_limit=time_limit)
        print(round(gap, digits = 1), "%\t") 
        print(res,round(gap, digits = 1), "%\t") 
        print("Erreurs train/test : ", prediction_errors(T,X_train,Y_train, classes))
        print(res,"Erreurs train/test : ", prediction_errors(T,X_train,Y_train, classes))
        print("/", prediction_errors(T,X_test,Y_test, classes), "\t")
        print(res,"/", prediction_errors(T,X_test,Y_test, classes), "\t")
        println(round(resolution_time, digits=1), "s")
        println(res,round(resolution_time, digits=1), "s")
        if !isMultivariate
            results[index]=string(gamma)*"&"*string(length(clusters))*"&"*string(round(resolution_time, digits=1))*"&"*string(round(gap, digits = 1))*"&"*string(prediction_errors(T,X_train,Y_train, classes))*"/"*string(prediction_errors(T,X_test,Y_test, classes))*"&"*string(nb)*"&"*string(nb_cons)*"&"
        else 
            results[index]=string(round(resolution_time, digits=1))*"&"*string(round(gap, digits = 1))*"&"*string(prediction_errors(T,X_train,Y_train, classes))*"/"*string(prediction_errors(T,X_test,Y_test, classes))*"&"*string(nb)*"&"*string(nb_cons)*"\\\\"
        end
        index+=1
    end
    println() 
    println(res)
    return results
end 
