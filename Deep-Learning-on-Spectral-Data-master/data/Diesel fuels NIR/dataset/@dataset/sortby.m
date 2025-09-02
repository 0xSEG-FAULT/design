function [B,index] = sortby(A,field,dim,set,direction)
%DATASET/SORTBY Sort a DataSet by given field, dim, and set.
% Allows sorting of a DataSet on any field associated with the data. 
%  INPUTS:
%    field - name of the field to sort.
%            'axisscale'
%            'class'
%            'classid'
%            'label'
%    dim   - dimension (mode) of field to use.
%    set   - set (column) to use.
%    direction - direction to sort, 'ascend' or 'descend'.
%
%I/O: [B,index] = sortby(A,field,dim,set,direction)
%I/O: [B,index] = sortby(A,'labels',1,2,'descend')
%
%See also: DATASET/SORTROWS

%Copyright Eigenvector Research, Inc. 2007-2019

if ~isa(A,'dataset') %redundant test for overloaded methods
  error('Function for DATASET ojects only.')
end
if nargin<2
  error('2 inputs required.')
end
if nargin<3
  dim = 1;
end
if nargin<4
  set = 1;
end
if nargin<5
  direction = 'ascend';
end

if ismember(field,{'axisscale' 'class' 'classid' 'label'})
  %Pull out vector and sort.
  indxstr = substruct('.',field,'{}',{dim set});
  myvec = subsref(A,indxstr);
  if isempty(myvec)
    error('Can''t sort on empty field.')
  end
  if ischar(myvec);
    %Change char array (labels) to cell array.
    myvec = str2cell(myvec);
  end
  
  if iscellstr(myvec)
    %Can't use direction when sorting cell of strings.
    [junk, index] = sort(myvec);
    if strcmp(direction,'descend')
      %Have to manually flip.
      index = index(end:-1:1);
    end
  else
    %Numeric column (class or axisscale) so sort with direction.
    [junk, index] = sort(myvec,direction);
  end
else
  error(['Can''t sort by "' field '" field.']);
end

%Call delsamps with sorted index to perform actuall sorting of DSO.
B = delsamps(A,index,dim,3);

thisname = inputname(1);
if isempty(thisname);
  thisname = ['"' A.name '"'];
end
if isempty(thisname);
  thisname = 'unknown_dataset';
end
caller = '';
try
  [ST,I] = dbstack;
  if length(ST)>1;
    [a,b,c]=fileparts(ST(end).name); 
    caller = [' [' b c ']'];
  end
catch
end

B.history = [B.history(1:end-1) {['x = sortby(' thisname ',''' field ''',' num2str(dim) ',' num2str(set) ',''' direction ''')   % ' timestamp caller ]}];


function testme()
%Temporary test commands.

%Make 3x3x3 cube in descending order.
a = dataset(reshape([27:-1:1],[3 3 3]));

%Create labels in ascending order.
a.label{1,1} = {'a' 'b' 'c'};
a.label{2,1} = {'2a' '2b' '2c'};
a.label{2,2} = {'22z' '22y' '22x'};

%Display first slab.
a.data(:,:,1)
%Display second slab.
a.data(:,:,2)

%Do a sort on mode 2/set 2 labels. Should reverse the order.
b = sortby(a,'label',2,2);

%Display first slab.
b.data(:,:,1)
%Display second slab.
b.data(:,:,2)

%-----------------------
%Try sorting with class.
a.class{3,2} = [1 2 3];

%Display first slab.
a.data(:,:,1)
%Display second slab.
a.data(:,:,2)

%Do a sort on mode 3/set 2 class. Should reverse slab order using descend.
b = sortby(a,'class',3,2,'descend');

%Display first slab.
b.data(:,:,1)
%Display second slab.
b.data(:,:,2)

