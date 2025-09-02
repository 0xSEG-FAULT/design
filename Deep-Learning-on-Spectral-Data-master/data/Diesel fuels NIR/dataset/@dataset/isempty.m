function out=isempty(in)
%DATASET/ISEMPTY Overload of standard isempty function for DataSet objects
%I/O: out = isempty(in)

% Copyright © Eigenvector Research, Inc. 2007-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.
%JMS

out = isempty(in.data);
