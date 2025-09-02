function edit(varargin)
%DATASET/EDIT Open a DataSet Object in the PLS_Toolbox DataSet Editor

%Copyright Eigenvector Research, Inc. 2000-2019

if exist('editds');
  targ = [];
  if nargin>1;
    targ = varargin{2};
    name = varargin{1};
  elseif nargin==1;
    targ = varargin{1};
    name = inputname(1);
  end
  h = editds(targ);  %create editor
  editds('setvarname',h,name);   %set name of variable
end
