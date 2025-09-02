function [B,index] = sortrows(A,column,direction)
%DATASET/SORTROWS Sort rows of a DataSet.
% Overload of the standard SORTROWS command. Input (column) allows
% specification of which rows should be used to sort. A negative value
% indicates that the given row should be sorted in decreasing order.
%
%I/O: [B,index] = sortrows(A,column)

%Copyright Eigenvector Research, Inc. 2007-2019

if nargin<3
  direction = 'ascend';
end

if nargin<2;
  [junk,index] = sortrows(A.data);
else
  if strcmp(direction,'descend')
    column = -column;
  end
  [junk,index] = sortrows(A.data,column);
end

S.type = '()'; S.subs = {index};
B = subsref(A,S);

thisname = inputname(1);
if isempty(thisname);
  thisname = ['"' varargin{1}.name '"'];
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
if nargin<2
  cmd = ['x = sortrows(' thisname ')'];
else
  cmd = ['x = sortrows(' thisname ',' num2str(column) ')'];
end
B.history = [B.history(1:end-1) { [cmd '  % ' timestamp caller ]}];
