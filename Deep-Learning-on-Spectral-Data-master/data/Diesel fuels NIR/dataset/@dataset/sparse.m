function out=sparse(in)
%DATASET/SPARSE Overload of standard sparse function for DataSet objects.
%I/O: out = sparse(in)

% Copyright © Eigenvector Research, Inc. 2007-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.
%JMS

out = sparse(in.data);
