function h = plot(dso,varargin)
%DATASET/PLOT Overload for plot command
% Simple plot command for DataSet objects. Plots columns of the data
% (unless number of columns is one) against an available axisscale. Include
% fields are applied prior to plotting with excluded data simply removed
% from matrix.
%
%I/O: h = plot(dso,...)
%I/O: h = plot(ax,dso,...)

% Copyright © Eigenvector Research, Inc. 2012-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.

if ~isa(dso,'dataset')
  if nargin<2 | ~isa(varargin{1},'dataset')
    error('First or second input must be a DataSet object')
  end
  ax = dso;
  dso = varargin{1};
  sz = size(ax);
  if all(sz>1)
    error('Input x must be a vector')
  end
  if max(sz)~=size(dso,1) & max(sz)==size(dso,2);
    %an x-axis which matches the COLUMNS of dso transposes dso (same as
    %with standard plot command)
    dso = dso';
  end
  varargin = varargin(2:end);
else
  ax = dso.axisscale{1};
end

if ~isempty(varargin)
  %check for other DSOs passed later (e.g. multiple data inputs)
  cl = cellfun('isclass',varargin,'dataset');
  if any(cl)
    error('PLOT only allows the first or second inputs to be DataSet objects')
  end
end

incl = dso.include(:,1);
if isempty(ax)
  h = plot(dso.data(incl{:}),varargin{:});
else
  if length(ax)<max(incl{1})
    error('x does not match size of DataSet y')
  end
  ax = ax(incl{1});  %apply include to x-axis values
  h = plot(ax,dso.data(incl{:}),varargin{:});
end

if nargout==0
  clear h
end
