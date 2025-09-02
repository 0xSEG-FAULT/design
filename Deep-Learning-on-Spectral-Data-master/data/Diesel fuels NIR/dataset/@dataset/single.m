function out = single(in)
%DATASET/SINGLE Convert DataSet object to a single precision array.
% Overload of the standard SINGLE command.
%I/O: out = single(in)

%Copyright Eigenvector Research, Inc. 2007-2019

out = single(in.data);
