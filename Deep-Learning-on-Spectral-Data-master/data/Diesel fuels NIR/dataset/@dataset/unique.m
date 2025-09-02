function varargout = unique(A,varargin)
%DATASET/UNIQUE Identify unique rows of a DataSet.
% Overload of the standard UNIQUE command. See UNIQUE help for information
% on the I/O available for this command.
% Note: UNIQUE operates based only on the data field of the DataSet object.
% Lables, axisscale, and classes are ignored during the unique command. In
% addition, only the last example of a given unique row is returned in
% the output unique DataSet (B).
%
%I/O: [B,I,J] = UNIQUE(A,...)

%Copyright Eigenvector Research, Inc. 2007-2019

%apply unique to DATA from dataset first...
[data,I,J] = unique(A.data,varargin{:});

if size(data,2)==1 && size(A,2)>1;
  %did not call unique with "rows" input. Matrix was reshaped into vector.
  %Drop all other dataset properties.
  B = dataset(data);
else
  %re-apply unique results to entire dataset
  S.type = '()';
  [S.subs{1:ndims(A)}] = deal(':');
  S.subs{1} = I;
  B = subsref(A,S);
end

varargout = {B,I,J};
