function out = numArgumentsFromSubscript(data,varargin)
%DATASET/NUMARGUMENTSFROMSUBSCRIPTS Returns the number of expected inputs 
% to subsasgn or the number of expected outputs from subsref. (always 1).
% Overload of any other numArgumentsFromSubscript command. Always returns a 
% value of 1 (one).
%
%I/O: out = numArgumentsFromSubscript(data, s,indexingContext)

%Copyright Eigenvector Research, Inc. 2007-2019

out = 1;
