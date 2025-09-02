function varargout = copy_clipboard(dso,varargin)
%DATASET/COPY_CLIPBOARD Copy data and labels to clipboard.
% Simple copy of (:,:,1) data, row labels, and column labels to system clipboard.
%   If nargout>0 then return table data without copying to clipboard. 
%
%I/O: copy_clipboard(dso)
%I/O: tdata = copy_clipboard(dso)

% Copyright © Eigenvector Research, Inc. 2013-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.

%TODO: Add flag for copying included data only.

if ~isa(dso,'dataset')
  error('First input must be a DataSet object')
end

tdata = dso.data(:,:,1);
tdata = num2cell(tdata);

%Column labels.
rowdata = str2cell(dso.label{1,1});
coldata = str2cell(dso.label{2,1})';

if ~isempty(rowdata)
  tdata = [rowdata tdata];
end

if ~isempty(coldata)
  if ~isempty(rowdata)
    %Add empty space for column of row names.
    tdata = [[{' '} coldata]; tdata];
  else
    tdata = [[coldata]; tdata];
  end
end

ttbl = cell2str(tdata,char(9),1)';
ttbl = [ttbl(:)]';

if nargout<1
  clipboard('copy',ttbl);
else
  varargout{1} = ttbl;
end
