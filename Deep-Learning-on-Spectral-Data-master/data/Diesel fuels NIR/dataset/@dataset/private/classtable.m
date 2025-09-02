function tbl = classtable(clsvec,mytbl)
%CLASSTABLE Creates a defualt class lookup table from a numeric class vector.
%  Returns a nx2 cell array with class number in first column and class
%  name in second column. If second input 'mytbl' given, then create new
%  table given existing table.
%I/O: tbl = classtable([1 1 2 2 3 2 1 2])

% Copyright © Eigenvector Research, Inc. 2005-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.
%RSK 11/02/2006

% clsvec = clsvec.class{mode,set};

%Can't use 'first', not available to 6.5.
%[cls junk clsnum] = unique(clsvec,'first'); %unique sorts so should always return empty at top if exists.

if nargin==1
  mytbl = '';
end

if isempty(mytbl)
  [cls junk clsnum] = unique(clsvec);
  tbl = [];
  if ~isempty(cls)
    for i = cls
      tbl = [tbl; num2cell(i) {['Class ' num2str(i)]}];
    end
  end
else
  tbl = mytbl;
  
  %Find classes not in existing table.
  newcls = unique(clsvec);
  newcls = newcls(~ismember(newcls,[mytbl{:,1}]));
  for i = 1:length(newcls)
    namenum = newcls(i);
    cpver = 0;
    while 1
      if cpver>0
        newname = ['Class ' num2str(namenum) '(' num2str(cpver) ')'];
      else
        newname = ['Class ' num2str(namenum)];
      end
      if ~ismember({newname},mytbl(:,2))
        break
      elseif cpver>1000
        %Break out of loop if more than 1000.
        break
      end
      cpver = cpver+1;
    end
    tbl = [tbl; {newcls(i)} {newname}];
  end
end
%Change "Class 0" to "unknown"

%Check for duplicates. This was happening when we used 'rand' to create
%large test class sets. The numbers were truncated when converted to
%strings and causing duplicates. This takes a while if there are a lot of
%classes, may need to vectorize somehow.
if ~isempty(tbl) && length(unique(tbl(:,2)))<size(tbl,1)
  my_dups = '';
  for i = unique(tbl(:,2))'
    if sum(ismember(tbl(:,2),i))>1
      my_dups = [my_dups i];
    end
  end
  
  for i = my_dups
    myloc = find(ismember(tbl(:,2),i));
    for j = 1:length(myloc)
      %Make name unique with letters but if something weird happesn and
      %we run out then use timestamp.
      if j<27
        mychar = char(96+j);
      else
        mychar = datestr(now,'yyyymmddTHHMMSS.FFF');
      end
      tbl(myloc(j),2) = {[tbl{myloc(j),2} '_' mychar]};
    end
  end
  
end




