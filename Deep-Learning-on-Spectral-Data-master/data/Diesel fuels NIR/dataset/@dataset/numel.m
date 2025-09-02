function out = numel(data,varargin)
%DATASET/NUMEL Return the total number of records in a DataSet (always 1).
% Overload of the standard NUMEL command. Always returns a value of 1
% (one).
%
% WARNING! Do NOT change this to return anything other than 1. Matlab does
% not correctly handle objects that return numel other than 1.
%
%I/O: out = numel(data)

%Copyright Eigenvector Research, Inc. 2007-2019

out = 1;
