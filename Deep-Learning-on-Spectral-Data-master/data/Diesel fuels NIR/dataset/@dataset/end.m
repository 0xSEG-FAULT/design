function val = end(x,vdim,ndim)
%DATASET/END Overloaded method for dataset objects.
%  Handle "end" keyword for dataset object indexing

%Copyright Eigenvector Research, Inc. 2002-2019
%JMS 9/9/02

val = size(x.data,vdim);

%The following code can be re-instated when subsref allows "incomplete
%indexing" (=reshaping) of an object such as:  mydataset(:)

% if vdim < ndim;
%   % (end,...)
%   val = size(x.data,vdim);
% else
%   % (...,end)  or  (end)
%   sz = size(x.data);
%   val = prod(sz(vdim:end));
% end
