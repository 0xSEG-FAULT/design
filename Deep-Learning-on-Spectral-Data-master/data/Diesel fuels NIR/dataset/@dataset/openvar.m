function openvar(varargin)
%DATASET/OPENVAR Open a DataSet Object in the DataSet Editor
% Overload to work with workspace double-click-to-edit

%Copyright Eigenvector Research, Inc. 2000-2019

if exist('editds');
  h = editds(varargin{2});  %create editor
  editds('setvarname',h,varargin{1});   %set name of variable
end
