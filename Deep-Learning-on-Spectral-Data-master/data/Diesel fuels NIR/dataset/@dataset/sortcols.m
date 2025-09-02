function [B,index] = sortcols(A,col,direction)
%DATASET/SORTCOLS Sort columns of a DataSet.
% Analog of the standard SORTROWS command. Input (col) allows
% specification of which columns should be used to sort. A negative value
% indicates that the given column should be sorted in decreasing order.
%
%I/O: [B,index] = sortcols(A,col)

%Copyright Eigenvector Research, Inc. 2007-2019

if nargin<3
  direction = 'ascend';
end

if nargin<2;
  [junk,index] = sortrows(A.data');
else
  if strcmp(direction,'descend')
    col = -col;
  end
  [junk,index] = sortrows(A.data',col);
end

S.type = '()'; S.subs = {':' index};
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
  cmd = ['x = sortcols(' thisname ')'];
else
  cmd = ['x = sortcols(' thisname ',' num2str(col) ')'];
end
B.history = [B.history(1:end-1) { [cmd '  % ' timestamp caller ]}];

