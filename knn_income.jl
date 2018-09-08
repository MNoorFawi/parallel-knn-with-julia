using DataFrames

println("### READING DATA ###")
adult_train = readtable("adult.data", separator = ',', header = false);
adult_test = readtable("adult.test", separator = ',', header = false, skipstart = 1);
println((:data_shape, size(adult_train), size(adult_test)))
println("")

println("# Preparing Data:")
println("1 Deleting any row containing any unknown value.")
trn = adult_train[.!vec(any(convert(Array, adult_train) .== "?", 2)), :];
tst = adult_test[.!vec(any(convert(Array, adult_test) .== "?", 2)), :];
println((:new_data_shape, size(trn), size(tst)))
println("")

println("2 one hot encoding string columns and normalizing numeric ones.")
function preprocess(new::DataFrame, old::DataFrame)
	dataType = describe(old)
	x = DataFrame()
	d = DataFrame()
	str = dataType[dataType[:eltype] .== String, :variable]
	num = dataType[(dataType[:eltype] .== Float64) .| (dataType[:eltype] .== Int64), :variable]
	str = setdiff(str, [names(old)[end]])
	for i in str
		dict = unique(old[:, i])
		for key in dict
			x[:, [Symbol(key)]] = map(Float32, 1.0(new[:, i] .== key))
		end
	end
	for i in num
		d[:, i] = map(Float32, (new[:, i]- minimum(new[:, i])) / (maximum(new[:, i]) - minimum(new[:, i])))
	end
	x = hcat(x, d)
	x[:y] = map(UInt8, (new[end] .== ">50K") .| (new[end] .== ">50K."))
	return x
end;

encoded_train = preprocess(trn, trn);
encoded_test = preprocess(tst, trn);
println((:encoded_data_shape, size(encoded_train), size(encoded_test)))
println("")

println("3 Splitting data into train and test")
xtrain, ytrain = [Array(encoded_train[1:end-1])', Array(encoded_train[end])];
xtest, ytest = [Array(encoded_test[1:end-1])', Array(encoded_test[end])];
println("# size of train and test data")
println(size(xtrain), size(xtest))
println("")

println("# Define the Model:")
println("* Adding 4 processes => define euclidean distance function")
println(" 	=> get k nearest neighbors => assign new labels")
addprocs(4);
println((:number_of_processes, nprocs()))

@everywhere function euclidean_distance(a, b)
	dist = 0.0 
	for i in 1:size(a, 1) 
		dist += (a[i] - b[i]) ^ 2
	end
	return sqrt(dist)
end;

@everywhere function KNN(x, i, k)
	nrows, ncols = size(x)
	a = Array{Float32}(nrows)
	for indx in 1:nrows
		a[indx] = x[indx, i]
	end
	b = Array{Float32}(nrows)
	dist = Array{Float32}(ncols) 
	for j in 1:ncols
		for indx in 1:nrows
			b[indx] = x[indx, j]
		end
		dist[j] = euclidean_distance(a, b)
	end
	sorted = sortperm(dist)
	knneighbors = sorted[2:k+1]
	return knneighbors
end;

@everywhere function assign_label(x, y, k, i)
	knn = KNN(x, i, k) 
	most_common = Array{Int64}
	Mean = mean(y[knn])
	if Mean >= 0.5
		most_common = 1
	else 
		most_common = 0
	end
	return most_common
end;

# a good k will be the one with 10 positive values 
prob_pos = sum(ytrain .== 1) / length(ytrain);
k = map(Int64, round(10 / prob_pos));
println("")

println("# Get predictions and accuracy on train and test data using $(k) k")
predictions = @parallel (vcat) for i in 1:size(xtrain, 2)
 assign_label(xtrain, ytrain, k, i)
end;
# calculate accuracy
correct = @parallel (+) for i in 1:size(xtrain, 2)
 assign_label(xtrain, ytrain, k, i) == ytrain[i]
end;
println((:train_accuracy, correct / length(ytrain)))
println("")

predictions = @parallel (vcat) for i in 1:size(xtest, 2)
 assign_label(xtest, ytest, k, i)
end;
# calculate accuracy
correct = @parallel (+) for i in 1:size(xtest, 2)
 assign_label(xtest, ytest, k, i) == ytest[i]
end;
println((:test_accuracy, correct / length(ytest)))
println("")


