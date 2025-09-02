function [tblnew, clsnew]= tablemerge(tbl1,tbl2)
%TABLEMERGE Merges two lookup tables.
%  The first table (tbl1) takes precedence if a numeric class has two
%  different names. If there is a duplicated name (with two different
%  numeric classes then "DuplicatedName" is appended to the tbl2 name.
%
%  Output 'tblnew' is a nx2 cell array in lookup table format. 
%
%  The second ouput 'clsnew' is the new class assignments (if any) for
%  duplicate numeric classes with differnt string values. It contains .str
%  the string name, .oldnum the old class numeric value, and .newnum the
%  new class numeric value. 
%
%I/O: [tbl clsnew] = tablemerge(mytable,myothertable)

% Copyright © Eigenvector Research, Inc. 2005-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.
%RSK 11/29/2006

%Thre cases:
% 1) Class name and class number are the same, do nothing.
% 2) Class name same but class number different, tbl2 number changes.
% 3) Class name differnt but number the same, tbl2 number changes.


tblnew = tbl1;
clsnew = '';
addcls = '';

%Find the max class number so if/when we create a new class number we know
%it won't be in either existing class vectors.
maxc = max([tbl1{:,1} tbl2{:,1}]);

for i = 1:size(tbl2,1)
  %Check to see if string class exists in first table. If it exists then it
  %has precedence so continue to next.
  myidx = min(find(ismember(tbl1(:,2),tbl2{i,2})));
  if ~isempty(myidx)
    %Duplicate name.
    if tbl1{myidx,1}==tbl2{i,1}
      %Case 1, names and numbers the same so do nothing.
      continue
    else
      %Case 2, same names but different numbers.
      clsnew(end+1).str = tbl2{i,2};
      clsnew(end).oldnum = tbl2{i,1};
      clsnew(end).newnum = tbl1{myidx,1};
      continue
    end

  else
    numidx = tbl2{i,1}==[tbl1{:,1}];
    if any(numidx)
      %Case 3, different name but duplicate number in tbl2 so move to new number.
      clsnew(end+1).str = tbl2{i,2};
      clsnew(end).oldnum = tbl2{i,1};
      maxc = maxc+1;
      clsnew(end).newnum = maxc;
      
      tbl2{i,1} = maxc;
    end
    tblnew = [tblnew;tbl2(i,:)];
  end
end

%Sort table based on numeric class.
nclass = [tblnew{:,1}]';
[sortednclass sindex] = sort(nclass);
strclass = [tblnew(:,2)];
sortedstrclass = strclass(sindex);
tblnew = [num2cell(sortednclass) sortedstrclass];


% 
% tblnew = tbl1;
% clsnew = '';
% addcls = '';
% for i = 1:size(tbl2,1)
%   %Check to see if numeric class exists in first table. If it exists then it
%   %has precedence so continue to next.
%   myidx = tbl2{i,1}==[tbl1{:,1}];%Index of numeric match.
%   if any(myidx)
%     %Numeric entry in tbl2 exists in tbl1 so check to see if string name is
%     %the same. If not, then need to append a new entry at end of table and
%     %output the new information so changes can be made to numeric class
%     %field (in calling function e.g., cat).
%     if ~strcmp(tbl2{i,2},tbl1{myidx,2})
%       %Now need to check if there's a duplicate name elsewhere in tbl1.
%       dupnameidx = ismember(tbl1(:,2),tbl2{i,2});
%       if any(dupnameidx)
%         clsnew(end+1).str = [tbl2{i,2} '_DuplicatedName'];
%       else
%         clsnew(end+1).str = tbl2{i,2};
%       end
%       clsnew(end).oldnum = tbl2{i,1};
%     end
%       continue
% 
%   elseif ismember(tbl2{i,2},tbl1(:,2))
%     tbl2{i,2} = [tbl2{i,2} '_DuplicatedName'];
%   end
%   tblnew = [tblnew;tbl2(i,:)];
% end
% 
% %Enter new classes for duplicate numerical values identified above.
% if ~isempty(clsnew)
%   maxc = max([tblnew{:,1}]);%Start adding new numeric classes based on max of new lutable.
%   for j = 1:length(clsnew)
%     tblnew = [tblnew; {maxc+j clsnew(j).str}];
%     clsnew(j).newnum = maxc+j;
%   end
% end
% 
% %Sort table based on numeric class.
% nclass = [tblnew{:,1}]';
% [sortednclass sindex] = sort(nclass);
% strclass = [tblnew(:,2)];
% sortedstrclass = strclass(sindex);
% tblnew = [num2cell(sortednclass) sortedstrclass];
% 
