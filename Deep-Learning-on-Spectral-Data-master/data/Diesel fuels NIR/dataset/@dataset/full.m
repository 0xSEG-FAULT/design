function out=full(in)
%DATASET/FULL Overload of standard full function for DataSet objects.
%I/O: out = full(in)

% Copyright © Eigenvector Research, Inc. 2007-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.
%JMS

out = full(in.data);
