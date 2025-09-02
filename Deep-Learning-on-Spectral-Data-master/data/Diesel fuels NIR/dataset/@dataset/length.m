function L = length(x)
%DATASET/LENGTH Length of DataSet object.
%  Returns length of data stored in a DataSet object. It is equivalent
%  to MAX(SIZE(X)) for non-empty arrays and 0 for empty ones. See built-in
%  Matlab function LENGTH for more information.
%  Input is the dataset of interest (x) and output is the length (L).
%I/O: L = length(x)

%Copyright Eigenvector Research, Inc. 2005-2019
%JMS 2/18/05

L = max(size(x));
