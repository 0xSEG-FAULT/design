function dsout = rmset(dsin,field,dim,set);
%RMSET Removes a specific set from a DSO field.
%  The function will check to see the field is of type cell and that the
%  dim/set exists then remove it. Any remaining sets will be shifted to
%  appropriately to fill the removed cell. If there are sets in different
%  dims that aren't empty then the set is just emptied along with
%  corresponding dependencies and not shifted.
%
%I/O: mydso = rmset(mydso,'axisscale',dim,set);
%I/O: mydso = rmset(mydso,'axisscale',1,3);
%
%See also: COPYDSFIELDS, DATASET, SQUEEZE

%Copyright Eigenvector Research, Inc. 2009-2019
%Licensee shall not re-compile, translate or convert "M-files" contained
% in PLS_Toolbox for use with any software other than MATLAB®, without
% written permission from Eigenvector Research, Inc.

if nargin < 4
  error('RMSET requires 4 inputs.')
end

%Check recursive.
if length(set)>1
  dsout = dsin;
  set = sort(set);
  %Need to do this in reverse order so sets are removed from end and
  %indexing isn't affected.
  for i = length(set):-1:1
    dsout = rmset(dsout,field,dim,set(i));
  end
  return
end

if strcmp(field,'classid')
  field = 'class';
end

%Note field names and dims for each.
myfieldnames = {'class' 3; 'classname' 3; 'axisscale' 3; 'axisscalename' 3;...
                'imageaxisscale' 3; 'imageaxisscalename' 3;...
                'label' 3; 'labelname' 3; 'classlookup' 2; 'title' 3; 'titlename' 3};

if ~ismember(field,myfieldnames(:,1))&&~ismember(field,fieldnames(dsin))
  error('Input field does not exist in dataset.')
end

%Create output.
dsout = dsin;

%Empty the field first.
switch field
  case 'class'
    fsize = checkfieldsize(dsin,'class',dim, set,'class',myfieldnames);
    %Clear class, classsname, and classlookup.
    dsout.class{dim,1,set} = [];
    dsout.class{dim,2,set} = '';
    dsout.classlookup{dim,set} = {};
    sfield = {'class' 'classlookup'};
  case 'classname'
    fsize = checkfieldsize(dsin,'class',dim, set,'classname',myfieldnames);
    dsout.class{dim,2,set} = '';
    sfield = {'class'};
  case 'axisscale'
    fsize = checkfieldsize(dsin,'axisscale',dim, set,'axisscale',myfieldnames);
    %Clear both.
    dsout.axisscale{dim,1,set} = [];
    dsout.axisscale{dim,2,set} = '';
    sfield = {'axisscale'};
  case 'axisscalename'
    fsize = checkfieldsize(dsin,'axisscale',dim, set,'axisscalename',myfieldnames);
    dsout.axisscale{dim,2,set} = '';
    sfield = {'axisscale'};
  case 'imageaxisscale'
    fsize = checkfieldsize(dsin,'imageaxisscale',dim, set,'imageaxisscale',myfieldnames);
    %Clear both.
    dsout.imageaxisscale{dim,1,set} = [];
    dsout.imageaxisscale{dim,2,set} = '';
    sfield = {'imageaxisscale'};
  case 'imageaxisscalename'
    fsize = checkfieldsize(dsin,'imageaxisscale',dim, set,'imageaxisscalename',myfieldnames);
    dsout.imageaxisscale{dim,2,set} = '';
    sfield = {'imageaxisscale'};
  case 'label'
    fsize = checkfieldsize(dsin,'label',dim, set,'label',myfieldnames);
    %Clear both.
    dsout.label{dim,1,set} = '';
    dsout.label{dim,2,set} = '';
    sfield = {'label'};
  case 'labelname'
    fsize = checkfieldsize(dsin,'label',dim, set,'lablename',myfieldnames);
    dsout.label{dim,2,set} = '';
    sfield = {'label'};
  case 'classlookup'
    fsize = checkfieldsize(dsin,'classlookup',dim, set,'classlookup',myfieldnames);
    dsout.classlookup{dim,set} = {};
    sfield = {'classlookup'};
  case 'title'
    fsize = checkfieldsize(dsin,'title',dim, set,'title',myfieldnames);
    %Clear both.
    dsout.title{dim,1,set} = '';
    dsout.title{dim,2,set} = '';
    sfield = {'title'};
  case 'titlename'
    fsize = checkfieldsize(dsin,'title',dim, set,'titlename',myfieldnames);
    dsout.titlename{dim,2,set} = '';
    sfield = {'title'};
  otherwise
    error(['Unrecognized field for rmset: ' field '.'])
end

%If every item is empty for given field AND there's data adjacent, then delete (column) and shift other
%sets over (to left). Otherwise, just clear the set and don't shift.
setempty = 1;
dimlist = 1:fsize(1);

for fld = sfield
  fdims = myfieldnames{ismember(myfieldnames(:,1),fld{:}),2};
  for i = dimlist
    if fdims==3
      %Sets are in mode 3.
      if ~isempty(dsout.(fld{:}){i,1,set}) || ~isempty(dsout.(fld{:}){i,2,set})
        setempty = 0;
      end
    elseif fdims==2 && ~isempty(dsout.(fld{:}){i,set})
      %Sets are in mode 2.
      setempty = 0;
    end
  end

  if setempty
    %Shift cells.
    if fdims==3 && size(dsout.(fld{:}),3)>1
      myfield = cat(3,dsout.(fld{:})(:,:,1:set-1),dsout.(fld{:})(:,:,set+1:end));
      dsout.(fld{:}) = cat(3,dsout.(fld{:})(:,:,1:set-1),dsout.(fld{:})(:,:,set+1:end));
    elseif fdims==2 && size(dsout.(fld{:}),2)>1
      dsout.(fld{:}) = cat(2,dsout.(fld{:})(:,1:set-1),dsout.(fld{:})(:,set+1:end));
    end
  end
end

thisname = inputname(1);
if isempty(thisname);
  thisname = ['"' dsin.name '"'];
end
if isempty(thisname);
  thisname = 'unknown_dataset';
end

caller = '';
try
  [ST,I] = dbstack;
  if length(ST)>2;
    [a,b,c]=fileparts(ST(3).name);
    caller = [' [' b c ']'];
  end
catch
end

[mytimestamp,dsout.moddate] = timestamp;   %and update moddate
notes  = ['   % ' mytimestamp caller];

dsout.history{end+1} = ['rmset(' thisname ',''' field ''',' num2str(dim) ',' num2str(set) ')' notes ];
%----------------------------------------------
function fsize = checkfieldsize(dsin,field,dim,set,fieldname,myfieldnames)
%Check to see if field exists.

%Get field size.
fsize = size(dsin.(field)); %Field size.
fdims = myfieldnames{ismember(myfieldnames(:,1),fieldname),2};
if fsize(1)<dim
  error(['Dimension (mode) does not exist for ' fieldname '.'])
end

if fdims == 3
  if length(fsize)<3
    %Only one set present, 3rd dim is singleton.
    if set~=1
      error(['Set does not exist for ' fieldname '.'])
    end
  elseif fsize(3)<set
    error(['Set does not exist for ' fieldname '.'])
  end
elseif fdims == 2 && fsize(2)<set
  error(['Set does not exist for ' fieldname '.'])
end


