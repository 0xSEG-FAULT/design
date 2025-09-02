function [liststr, tblidx] = listsets(dso_in,fieldname,fieldmode)
%DATASET/SETLIST For a given field and mode list the sets available.
%  If field name/mode empty then list all. Second output (tblidx) is a map
%  of list value back to dataset index with columns (Value Field Mode Set)
%  where mode and or set = 0 for header lines.
%
%  Table (tblidx) has columns:
%    index      field name      mode      set
%
%  Output is indented list:
%
%  liststr=
%     LABEL                      
%       Mode 1                   
%         Quarry  [26]           
%       Mode 2                   
%         Element  [17]          
%     CLASS                      
%       Mode 1                   
%         Quarry  [5]            
%         Set 2  [empty]         
%         Set 3  [5]             
%     ...
%
%
%  tblidx=
%       [ 1]    'label'        [0]    [0]
%       [ 2]    'label'        [1]    [0]
%       [ 3]    'label'        [1]    [1]
%       [ 4]    'label'        [2]    [0]
%       [ 5]    'label'        [2]    [1]
%       [ 6]    'class'        [0]    [0]
%       [ 7]    'class'        [1]    [0]
%       [ 8]    'class'        [1]    [1]
%       [ 9]    'class'        [1]    [2]
%       [10]    'class'        [1]    [3]
%     ...
%
%I/O: out = listsets(dso_in,fieldname,fieldmode)

% Copyright © Eigenvector Research, Inc. 2012-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.

if isempty(fieldname)
  fieldname = {'label' 'class' 'axisscale'};
end

if ~iscell(fieldname)
  fieldname = {fieldname};
end

if isempty(fieldmode)
  fieldmode = 1:ndims(dso_in);
end

liststr = '';
%Index map of list value to dso.
% Value Field Mode Set
% 3     class 2    1  
tblidx = {};

for i = 1:length(fieldname)
  liststr = char(liststr,['' upper(fieldname{i})]);
  tblidx = [tblidx;{size(liststr,1)-1 fieldname{i} 0 0}];
  fdata = dso_in.(fieldname{i});
  fsize = size(fdata);
  fsets = 1;
  if length(fsize)>2
    fsets = fsize(3);
  end
  for j = 1:length(fieldmode)
    if length(fieldmode)>1
      %Only add MODE if more than one is being listed.
      liststr = char(liststr,[' MODE ' num2str(fieldmode(j))]);
      tblidx = [tblidx;{size(liststr,1)-1 fieldname{i} fieldmode(j) 0}];
    end
    for k = 1:fsets
      nm = fdata{fieldmode(j),2,k};
      if strcmpi(fieldname{i},'label');
        fsz = num2str(size(unique(fdata{fieldmode(j),1,k},'rows'),1));
      else
        fsz = num2str(size(unique(fdata{fieldmode(j),1,k}),2));
      end
      
      if strcmp(fsz,'0')
        fsz = '(empty)';
      else
        fsz = ['(' fsz ')'];
      end
      
      if isempty(nm)
        nm = ['Set ' num2str(k)];
      end
      liststr = char(liststr,['  ' nm '  ' fsz]);
      tblidx = [tblidx;{size(liststr,1)-1 fieldname{i} fieldmode(j) k}];
    end
  end
end

liststr = liststr(2:end,:);%Get rid of first empty line.
