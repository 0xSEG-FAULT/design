function varargout = times(varargin)
%DATASET/TIMES Overload of the array multiply .* operator for DataSet Objects

% Copyright © Eigenvector Research, Inc. 2006-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.
%JMS

if nargin>2
  error(['Too many input arguments'])
end
if nargin<2
  error(['Not enough input arguments'])
end

if prod(size(varargin{1}))>1 &  prod(size(varargin{2}))>1
  if ndims(varargin{1})~= ndims(varargin{2}) | any(size(varargin{1})~=size(varargin{2}))
    error('Matrix dimensions must agree');
  end
elseif prod(size(varargin{1}))==1
  %replicate item 1 to match size of item 2 (so labels work out
  varargin{1} = repmat(varargin{1},size(varargin{2}));
elseif prod(size(varargin{2}))==1
  %replicate item 2 to match size of item 1 (so labels work out
  varargin{2} = repmat(varargin{2},size(varargin{1}));
end

for j=1:2
  if ~isa(varargin{j},'dataset')
    varargin{j} = dataset(varargin{j});
  end
end

%do actual operation
out = varargin{1};
out.data = varargin{1}.data .* varargin{2}.data;

name = '';
for ii=1:2;
  thisname = inputname(ii);
  if isempty(thisname);
    thisname = ['"' varargin{ii}.name '"'];
  end
  if isempty(thisname);
    thisname = 'unknown_dataset';
  end
  name = [name '.*' thisname];
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
out.history = [out.history {['x = ' name(3:end) '   % ' timestamp caller ]}];

out = mergelabels(out,varargin{2},'*');  %combine labels in all modes

varargout = {out};
