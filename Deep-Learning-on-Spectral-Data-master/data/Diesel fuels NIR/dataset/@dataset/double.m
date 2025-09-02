function out = double(in)
%DATASET/DOUBLE Convert DataSet object to a double precision array.
% Overload of the standard DOUBLE command.
%I/O: out = double(in)

%Copyright Eigenvector Research, Inc. 2007-2019

out = double(in.data);
