function varargout = uminus(varargin)
%DATASET/TIMES Overload of the unary minus operator for DataSet Objects

% Copyright © Eigenvector Research, Inc. 2006-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.
%RSK

if nargin>1
  error(['Too many input arguments'])
end
if nargin<1
  error(['Not enough input arguments'])
end

if ~isa(varargin{1},'dataset')
  varargin{1} = dataset(varargin{1});
end

%do actual operation
out = varargin{1};
out.data = varargin{1}.data .* -1;

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
out.history = [out.history {['x = ' thisname '-1   % ' timestamp caller ]}];

varargout = {out};
