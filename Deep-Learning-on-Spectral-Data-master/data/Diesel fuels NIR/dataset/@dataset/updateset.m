function [dso,useset] = updateset(dso,field,mode,value,name,classlookup)
%DATASET/UPDATESET Add/update a label field (axisscale,label,class) in a DataSet.
% Locates a class, axisscale, or label set, as indicated in the "field"
% input, with the matching name (if name is supplied) and updates it with
% the value. If no name is supplied or no matching name is found in the
% DataSet, the new values are added as a new set. Also assigns fieldname
% and lookup table (if supplied for classes) to the set.
%
% If "name" is omitted, the values will always be added into the first
% empty set or onto the end of all sets (if no empty sets found).
% If "name" is supplied, any set with the corresponding field name
% (classname, axisscalename, labelname) will be overwritten with the value
% or a new set will be created to hold the values. 
%
% Outputs are the modified DataSet (dso) and the set that was modified or
% added (useset).
%
%I/O: [dso,useset] = updateset(dso,'field',mode,value)              %new unnamed class
%I/O: [dso,useset] = updateset(dso,'class',mode,class,classlookup)  %new unnamed class
%
%I/O: [dso,useset] = updateset(dso,'field',mode,value,name)             %update/add
%I/O: [dso,useset] = updateset(dso,'class',mode,class,name,classlookup) %update/add with class looup
%
%See also: DATASET/SUBSASGN

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

%figure out what they gave us WRT the lookup and name fields
no_classlookup = 0;%Flag if called with no lookup.
if nargin<5
  no_classlookup = 1;
  classlookup = {};
  name = '';
elseif nargin==5
  if iscell(name)
    %only a lookup table, swap lookup and name
    classlookup = name;
    name = '';
  else
    no_classlookup = 1;
    classlookup = {};
  end
elseif nargin>5
  if iscell(name)
    %bad user - assume they swapped
    temp = name;
    name = classlookup;
    classlookup = temp;
  end
end

%test field name
field = lower(field);
if ~ismember(field,{'class','axisscale','label' 'title'});
  error('Invalid field to modify "%s"',field);
end

%get current sets for the mode we need
sets = dso.(field);
names = squeeze(sets(mode,2,:));
sets  = squeeze(sets(mode,1,:));

useset = 0;
if ~isempty(name)
  %look for a set matching this name
  for j=1:length(names);
    if strcmpi(name,names{j})
      useset = j;
      break;
    end
  end
end
if useset==0
  %look for first empty set
  for j=1:length(sets);
    if isempty(sets{j})
      useset = j;
      break;
    end
  end
end
if useset==0;
  %no empty sets found, use end
  useset = length(sets)+1;
end

%add the classes and clasname (and classlookup table if we have it)
dso = subsasgn(dso,substruct('.',[field 'name'],'{}',{mode useset}),name);
if ~no_classlookup & strcmp(field,'class')
  %do this BEFORE assigning class (in case the lookup doesn't completely
  %match the IDs supplied)
  dso = subsasgn(dso,substruct('.','classlookup','{}',{mode useset}),classlookup);
end
dso = subsasgn(dso,substruct('.',field,'{}',{mode useset}),value);
