function setnum = findset(dso,field,mode,name)
%DATASET/FINDSET Locate a set within a label field (axisscale,label,class) in a DataSet.
% Locates a class, axisscale, or label set, as indicated in the "field"
% input, on the indicated mode, with the matching name (if name is
% supplied).
%
% Output is the set number of the first matching set.  If no matching set
% is found, an empty matrix is returned.
%
%I/O: setnum = findset(dso,'field',mode,name)
%
%See also: DATASET/SUBSASGN, DATASET/UPDATESET

%Copyright Eigenvector Research, Inc. 2012-2019
%Licensee shall not re-compile, translate or convert "M-files" contained
% in PLS_Toolbox for use with any software other than MATLAB®, without
% written permission from Eigenvector Research, Inc.

if nargin<4;
  error('Incorrect number of inputs')
end
if ~isa(dso,'dataset')
  dso = dataset(dso);
end

%test field name
field = lower(field);
if ~ismember(field,{'class','axisscale','label' 'title'});
  error('Invalid field to modify "%s"',field);
end

%get current sets for the mode we need
sets = dso.(field);
names = squeeze(sets(mode,2,:));

setnum = [];
%look for a set matching this name
for j=1:length(names);
  if strcmpi(name,names{j})
    setnum = j;
    break;
  end
end

