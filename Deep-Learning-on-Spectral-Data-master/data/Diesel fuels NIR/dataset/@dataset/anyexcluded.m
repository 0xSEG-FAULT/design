function out = anyexcluded(data,dim)
%ANYEXCLUDED Returns true if any element of the given DSO is excluded.
% This test reports if any item in the given DataSet object (data) has been
% excluded.
%
% Optional input (dim) can be a single index or vector of indexes
% indicating which dimensions of the dataset should be tested for
% exclusions. For example:
%    anyexcluded(data,[1 3])
% reports if there are any exclusions on dimensions 1 or 3.
%
%I/O: out = anyexcluded(data)
%I/O: out = anyexcluded(data,dim)

%Copyright Eigenvector Research, Inc. 2003-2019

%get size of include and data
sz_include = cellfun('length',data.include);
sz_data = size(data.data);

if length(sz_data)<length(sz_include); 
  %handle special case of singleton dimensions (which don't show up in
  %data.data size, but do show up in include size)
  sz_data(end+1:length(sz_include)) = 1; 
end

%do test
if nargin<2
  dim = 1:length(sz_include);
end
dim(dim>length(sz_include) | dim<1) = []; %drop dims which don't exist (and are, by definition, not excluded)

if ndims(sz_data)~=ndims(sz_include) | any(sz_data(dim)~=sz_include(dim))
  out = true;
else
  out = false;
end
  
