include("building_tree_.jl")
include("utilities.jl")
include("read_data.jl")

function main(inegalite::Bool=false,save::Bool=true)

    # Pour chaque jeu de données
    for dataSetName in ["iris", "seeds", "wine","dry_bean","mushroom"]
        if inegalite &&save
            res=open("..//results-inegalites-"*dataSetName*".txt","w")
            res_save=open("..//save-results-inegalites-"*dataSetName*".txt","w")
        elseif inegalite && !(save)
            res=open("..//results-inegalites-"*dataSetName*".txt","w")
        elseif save && !inegalite 
            res_save=open("..//save-results-"*dataSetName*".txt","w")
            res=open("..//results-"*dataSetName*".txt","w")
        else 
            res=open("..//results-"*dataSetName*".txt","w")
        end
        if save 
            println(res_save,"\\begin{tabular}{cc||c||ccccc|ccccc}")
            println(res_save,"Dataset&",dataSetName,"&D&\\multicolumn{5}{c}{Univarié}&\\multicolumn{5}{c}{Multivarié}\\\\ \\hline")
        end
        print("=== Dataset ", dataSetName)

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
        Y_train = Y[train]
        X_test = reducedX[test, :]
        Y_test = Y[test]
        classes = unique(Y)

        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2), ")")
        
        # Temps limite de la méthode de résolution en secondes
        time_limit = 1000
        if inegalite 
            ch=", Avec inégalités valides"
        else 
            ch=""
        end
        println(" (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2),", time limit: ", time_limit,ch, ")")
        println(res," (train size ", size(X_train, 1), ", test size ", size(X_test, 1), ", ", size(X_train, 2), ", features count: ", size(X_train, 2),", time limit: ", time_limit,ch, ")")
        if save 
            println(res_save,"Train size &",size(X_train, 1),"&&Solve time& Gap(\\%)& Errors train/test&Node count&Constraint count&Solve time& Gap(\\%)& Errors train/test&Node count&Constraint count\\\\\\cline{4-13}")
        end
        
        results=["Test size&"*string(size(X_test, 1))*"&","Features count&"*string(size(X_train, 2))*"&","Time limit &"*string(time_limit)*"&"]
        # Pour chaque profondeur considérée
        for D in 2:4

            print("  D = ", D)
            println(res,"  D = ", D)
            results[D-1]=results[D-1]*string(D)*"&"
            ## 1 - Univarié (séparation sur une seule variable à la fois)
            # Création de l'arbre
            print("    Univarié...  \t")
            T,obj,resolution_time,gap,nb,nb_cons= build_tree(X_train, Y_train, D,  classes, multivariate = false, time_limit = time_limit, inegalite = inegalite)
            # Test de la performance de l'arbre
            print(round(resolution_time, digits = 1), "s\t")
            print(res,round(resolution_time, digits = 1), "s\t")
            results[D-1]=results[D-1]*string(round(resolution_time, digits = 1))*"&"
            print("gap ", round(gap, digits = 1), "%\t")
            print(res,"gap ", round(gap, digits = 1), "%\t")
            results[D-1]=results[D-1]*string(round(gap, digits = 1))*"&"
            if T != nothing
                print("Erreurs train/test ", prediction_errors(T,X_train,Y_train, classes))
                print("/", prediction_errors(T,X_test,Y_test, classes), "\t")
                print(res,"/", prediction_errors(T,X_test,Y_test, classes), "\t")
                results[D-1]=results[D-1]*string(prediction_errors(T,X_train,Y_train, classes))*"/"*string(prediction_errors(T,X_test,Y_test, classes))*"&"
            else 
                results[D-1]=results[D-1]*"&&"
            end
            results[D-1]=results[D-1]*string(nb)*"&"*string(nb_cons)*"&"
            println("node count ",nb,"\t constraint number ",nb_cons)
            println(res,"node count ",string(nb),"\t constraint number ",nb_cons)
            ## 2 - Multivarié
            print("    Multivarié...\t")
            T, obj, resolution_time, gap,nb,nb_cons = build_tree(X_train, Y_train, D,  classes, multivariate = true, time_limit = time_limit,inegalite=inegalite)
            print(round(resolution_time, digits = 1), "s\t")
            print(res,round(resolution_time, digits = 1), "s\t")
            results[D-1]=results[D-1]*string(round(resolution_time, digits = 1))*"&"
            print("gap ", round(gap, digits = 1), "%\t")
            print(res,"gap ", round(gap, digits = 1), "%\t")
            results[D-1]=results[D-1]*string(round(gap, digits = 1))*"&"
            if T != nothing
                print("Erreurs train/test ", prediction_errors(T,X_train,Y_train, classes))
                print(res,"Erreurs train/test ", prediction_errors(T,X_train,Y_train, classes))
                print("/", prediction_errors(T,X_test,Y_test, classes), "\t")
                println(res,"/", prediction_errors(T,X_test,Y_test, classes), "\t")
                results[D-1]=results[D-1]*string(prediction_errors(T,X_train,Y_train, classes))*"/"*string(prediction_errors(T,X_test,Y_test, classes))*"&"
            else 
                results[D-1]=results[D-1]*"&&"
            end
            results[D-1]=results[D-1]*string(nb)*"&"*string(nb_cons)*"\\\\"
            println("node count  ",nb,"\t constraint number ",nb_cons)
            println(res,"node count "*string(nb),"\t constraint number ",nb_cons)
            if save 
                println(res_save,results[D-1])
            end
        end
        if save 
            println(res_save,"\\end{tabular}")
            close(res_save)
        end
        close(res)
    end 
end
