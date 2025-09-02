function [tbl,classvec]= stringtable(clsval,mytbl,tblreset)
%STRINGTABLE Creates a defualt class lookup table and numeric class vector from a cell array of strings.
%  Inputs are clsval (cell array of strings), mytbl (nx2 lookup table), and
%  tblreset ('yes' 'no') governs wether or not to rebuild a lookup table or
%  merge values to existing 'mytbl' input. Outputs 'tbl' as a nx2 cell
%  array with class number in first column and class name in second column.
%  Returns 'classvec' as a numeric vector of
%  positbve integers.
%I/O: [tbl classvec] = classtable({'class 1' 'class 1' 'class 3' 'class 2' 'Class 2})
%I/O: [tbl classvec] = classtable({'class 1' 'class 1' 'class 3' 'class 2' 'Class 2},mytbl)

% Copyright © Eigenvector Research, Inc. 2005-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.
%RSK 11/02/2006

if nargin == 1
  %No lookup table supplied.
  mytbl = [];
end

%Default table reset value set below.

%If there is an existing lookup table:
%  Use these values to build class set to preserve plotting behavior. Add
%  new classes as needed.
%
%Look for existing Class 0, if there is one then assign only elements
%  to it matching its name. If there is no Class 0, then look for "unknown"
%  or '' and assign accordingly.
%
%If there is no existing lookup table:
%  Build table following Class 0 rules, elements labeled "unknown" become
%  Class 0 otherwise elements labled '' (empty) become Class 0. See naming
%  code below for exact precedence.
%

%Check is there is Class 0 in lookup table.
if ~isempty(mytbl)
  %Position of zero in lookup table.
  posz = find(ismember([mytbl{:,1}],0)); 
else
  posz = [];
end

%Get list of values and indexes.
[cls junk clsnum] = unique(clsval);
if ischar(cls)
  cls = {cls};
end

%If there is a Class 0 in the incoming table then use that name as the
%Class 0 in the incoming cell of strings (clsval).
%If there is no Class 0 in the incoming table then check for reserved words
%in the incoming table, remove any that are present, then assign Class 0
%based on naming precedence of key words. This will allow for the case
%where a user may assign a reserved word to a class other than 0 (e.g.,
%'unknown').
if ~isempty(posz)
  srchstr = mytbl(posz,2);
else
  %This list represents the order of precedence, the first string found is
  %the zero class. 
  srchstr = {'Class 0' 'class 0' 'Unknown' 'unknown' ''};
  if ~isempty(mytbl)
    zuse = ismember(srchstr,mytbl(:,2))==0;
    srchstr = srchstr(zuse);
  end
end

%Find positions of Class 0, unknown or empty values.
%The rest will end up in the list in alphabetical order.
for i = srchstr
  posu = find(ismember(cls,i));
  if ~isempty(posu)
    break
  end
end

%Only envoke zero naming code below if there is no existing Class 0.
if isempty(posz)
  %Make 0 entry into numeric class and move all other classes appropriately.
  if ~isempty(posu)
    %Change (posu) to 0.
    uindx = find(ismember(clsnum,posu));
    mindx = find(clsnum>posu);
    clsnum(uindx) = 0;
    clsnum(mindx) = clsnum(mindx)-1;
    %Move zero class to beginning of list.
    cls = [cls(posu) cls(find(~ismember(cls,cls{posu})))];
  end
end

if nargin < 3
  %Default table reset value.
  if isempty(mytbl)% || (isempty(intersect(mytbl(:,2),cls')) && length(clsval)~=1)
    tblreset = 'yes';
  else
    tblreset = 'no';
  end
end

%Create new lookup table. 
if strcmp(tblreset,'yes')
  %Create all new table and new class for given array of stings. If the new
  %class values don't exist in the existing table then "reset" the table.
  %Because the table is new we don't need to worry about what numbers are
  %assigned to which classes so use 'unique' to sort clsnum. The "unknown"
  %elements will be in the correct order from above. 
  cclsnum = num2cell(unique(clsnum));
  cclsnum = cclsnum(:);  %force column
  tbl = [cclsnum cls'];
else
  %Need to assign class number based on existing table then augment
  %on new values.
  
  %If there is no existing Class 0 AND there is a Class 0 in clsval then
  %spoof Class 0 into mytbl.
  if isempty(posz) && ~isempty(posu)
    mytbl = [0 cls(1); mytbl];
  end
  
  for ic = 1:size(mytbl,1)
    %Loop through existing class IDs and assign class number
    cinds = ismember(clsval,mytbl{ic,2});%Indices of sting class.
    clsnum(cinds) = mytbl{ic,1};%Change numeric class to existing class set.
  end
  remcls = setdiff(cls, mytbl(:,2));%Remaining new classes.
  for ic = 1:length(remcls)
    cinds = ismember(clsval,remcls{ic});%Indices of sting class.
    newc = max([mytbl{:,1}])+1;%New class number.
    clsnum(cinds) = newc;%Build numeric class set.
    mytbl = [mytbl; {newc remcls{ic}}];%Add new record to lookup table.
  end
  tbl = mytbl;
end
%Assign numeric values to output.
classvec = clsnum;





