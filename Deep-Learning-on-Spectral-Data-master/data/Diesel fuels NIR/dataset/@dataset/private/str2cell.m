function parsed = str2cell(str,linebreak_only)
%STR2CELL convert from string to cell array (using tab and lf chars)
% A multi-line string or a string with linebreak and/or tab
% characters is parsed into a cell array based on these separators.
% Input is a string or string array (str) and output is a cell array
% (parsed). If input is omitted, the contents of the clipboard are
% converted to a cell. Optional input (linebreak_only) limits conversion to
% linebreaks to create a column vector of cells containing strings for each
% line.
%
%I/O: parsed = str2cell(str,linebreak_only)
%
%See also: CELL2STR

% Copyright © Eigenvector Research, Inc. 2005-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.
%jms 3/2005

parsed    = {};
if nargin<1;
  str = clipboard('paste');
end

if iscell(str)
  parsed = str;
  return
end

if ~isstr(str);
  error('Input (str) or contents of clipboard must be a string or string array');
end

if nargin<2
  linebreak_only = 0;
end

oneline = size(str,1)==1;  %all as one line?

if oneline;
  %string is all one line
  linebreak = [0 findstr(str,10)];  %locate line breaks
  
  %Try looking for carriage return (especailly if you're on a Mac).
  if length(linebreak)==1
    linebreak = [0 findstr(str,13)];
  end
  
  if isempty(linebreak) | linebreak(1)>1;
    linebreak = [0 linebreak];
  end
  if linebreak(end) ~= length(str);
    linebreak = [linebreak length(str)+1];
  end
  nlines = length(linebreak)-1;
else
  %string is string array (already multi-line)
  nlines = size(str,1);
  linebreak = 1:nlines;
end

parsed = cell(nlines,1);
if linebreak_only
  %MUCH faster way of handling it
  for j = 1:nlines;
    if oneline
      temp = str((linebreak(j)+1):(linebreak(j+1)-1));
    else
      temp = str(j,:);
    end
    parsed{j,1} = temp;
  end
else
  %parsing out columns as well as lines... slower but more useful
  for j = 1:nlines;
    if oneline
      temp = str((linebreak(j)+1):(linebreak(j+1)-1));
    else
      temp = str(j,:);
    end
    if isempty(temp);
      parsed{j,1} = '';
      continue;
    end
    tab  = findstr(temp,9);   %locate tabs
    if isempty(tab) | tab(1)>1;
      tab = [0 tab];
    end
    if tab(end) ~= length(temp)
      tab = [tab length(temp)+1];
    end
    for k = 1:length(tab)-1;
      onecell = deblank(temp((tab(k)+1):(tab(k+1)-1)));
      if isempty(onecell); onecell = ''; end
      parsed{j,k} = onecell;
    end
  end
end
parsed = deblank(parsed);
