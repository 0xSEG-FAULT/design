function [value,idx] = findclassmatch(x,feyld,indexonly)
%FINDCLASSMATCH Returns the data and index associated with a classID of a Dataset
% x is a dataset object, feyld is the name to look for, indexonly is a
% boolean which, if true, only returns the "idx" avoiding the memory
% associated with extracting the content. Similar to FINDLABELMATCH.
%
%I/O: [value,idx] = findclassmatch(x,feyld,indexonly)
%I/O: [value,idx] = findclassmatch(x,feyld,indexonly)
%
%Copyright Eigenvector Research, Inc. 2010-2019
%Licensee shall not re-compile, translate or convert "M-files" contained
% in PLS_Toolbox for use with any software other than MATLAB®, without
% written permission from Eigenvector Research, Inc.

if nargin<3
  indexonly = false;
end

idx = [];
match = [];
clmatch = [];

% %look at classes for match to this field name
  for m = [2 1 3:ndims(x.data)];
    for set = 1:size(x.classlookup,2);
      lookup = x.classlookup{m,set};
      if ~isempty(lookup);
        clmatch = strmatch(feyld,lookup(:,2),'exact');
        if ~isempty(clmatch)
          match = find(x.class{m,1,set}==x.classlookup{m,set}{clmatch,1});
          %First class hit takes precedence even if there are no memebers.
          break
        end
      end
      if ~isempty(match); break; end
    end
    if ~isempty(match); break; end
  end

if ~isempty(clmatch)
  %got class match? get that part of the DSO
  idx.type = '()';
  [idx.subs{1:ndims(x.data)}] = deal(':');
  idx.subs{m} = match;
end

% - - - - - - - - - - - - - - 
%other tests

if isempty(idx)
  %look for class set NAME matches
  for m = [2 1 3:ndims(x.data)];
    match = strmatch(feyld,squeeze(x.class(m,2,:)),'exact');
    if ~isempty(match)
      idx   = struct('type',{'.' '{}'},'subs',{'classid' {m,match}});
      break;
    end
  end
end

if indexonly
  value = idx;
else
  if ~isempty(idx)
    value = subsref(x,idx);
  else
    value = [];
  end
end
