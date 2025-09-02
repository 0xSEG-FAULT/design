function n = ndims(x,dim)
%DATASET/NDIMS Number of dimensions of DataSet object.
%  Returns number of dimensions of the data stored in a DataSet object. See
%  built-in Matlab function NDIMS for more information Input is the dataset
%  of interest (x). Output is (n) the number of dimensions.
%I/O: n = ndims(x)

%Copyright Eigenvector Research, Inc. 2003-2019
%JMS 4/15/03
%jms 11/7/03 -revised to base on number of modes shown in include field

n = size(x.include,1);
% n = length(size(x));
  
