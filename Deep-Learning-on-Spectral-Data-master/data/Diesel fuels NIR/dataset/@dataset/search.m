function out = search(dso_in,fieldname,fieldmode,fieldset,searchterm)
%DATASET/SEARCH Search for given term in a dso field, mode, and set.
% Return 'out' as index of matches. Searches are case insensitive (use ""
% for case sensitive).
%
% NOTE: If field is numeric then 'searchterm' will be interpreted as
% numeric expression. First characters can be relational operators.
%
% NOTE: If fieldname is "index" then linear index is searched with numeric
% terms.
%
% Search Criteria (TEXT):
%
%  *   = wildcard (similar to SQL wildcard, not strict RE)
%  ""  = exact match.
%  re: = regular expression.
%  ml: = Matlab expression. [NOT IMPLEMENTED]
%
%I/O: out = search(dso_in,fieldname,fieldmode,fieldset,searchterm)

% Copyright © Eigenvector Research, Inc. 2012-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.

out = [];

if strcmpi(fieldname,'index')
  %Use index as numeric data.
  sz = size(dso_in);
  searchdata = 1:sz(fieldmode);
else
  searchdata = get(dso_in,fieldname,fieldmode,fieldset);
  if ~isnumeric(searchdata)
    if ischar(searchdata)
      searchdata = str2cell(searchdata);
    end
  end
end

%Parse search term.
termtype = '';
if length(searchterm)>3
  termtype = lower(searchterm(1:3));
  if ~ismember(termtype,{'re:' 'ml:' 'x=:'})
    termtype = '';
  else
    searchterm(1:3) = [];
  end
end

%Parse numeric expression.
if isnumeric(searchdata)
  out = searchnum(searchdata, searchterm);
else
  %Text search.
  switch termtype
    case 're:'
      out = regexp(searchdata,searchterm);
      if ~isempty(out)
        out = ~cellfun('isempty',out);
      else
        out = [];
      end
    case 'ml:'
      %NOT IMPLEMENTED
    otherwise
      %Parse exact term.
      if ~isempty(searchterm)&strcmp(searchterm(1),'"')&strcmp(searchterm(end),'"')
        %Case sensitive.
        searchterm = searchterm(2:end-1);
        out = strfind(searchdata, searchterm);
      else
        starloc = strfind(searchterm,'*');
        if ~isempty(starloc)
          searchterm = strrep(searchterm,'*','.*');
          
          %Anchor beginning and or end if needed.
          if starloc(1)~=1
            %Anchor beginning. Use this so you can search for "samples
            %begining with 'SH'"
            searchterm = ['^' searchterm];
          end
          
          if starloc(1)~=length(searchterm)
            %Anchor end. Use this so you can match instances of "samples ending
            %in .jpg..."
            searchterm = [searchterm '$'];
          end
          
          out = regexp(lower(searchdata),lower(searchterm));
        else
          %Case insensitive.
          out = strfind(lower(searchdata), lower(searchterm));
        end
      end
      if ~isempty(out)
        out = ~cellfun('isempty',out);
      else
        out = [];
      end
  end
end

%-----------------------
function out = searchnum(sdata, sterm)
%Search for numeric data.

sterm = strrep(lower(sterm),'and','&');
sterm = strrep(lower(sterm),',','|');
sterm = strrep(lower(sterm),'or','|');

loc = [0 find(ismember(str2cell(sterm(:)),{'|' '&'}))' length(sterm)+1];
out = [];
for i = 1:length(loc)
  if i == length(loc);
    break
  end
  
  %Isolate searh term.
  myterm = strtrim(sterm(loc(i)+1:loc(i+1)-1));
  
  if isempty(myterm)
    continue
  end
  %Get data for term.
  thissearch = searchterm(sdata, myterm);
  
  if isempty(thissearch)
    continue
  end
  
  %Combine searched data.
  if loc(i)==0
    %First search.
    out = thissearch;
  else
    switch sterm(loc(i))
      case '&'
        if ~isempty(out)
          out = and(out,thissearch);
        end
      case '|'
        if ~isempty(out)
          out = or(out,thissearch);
        else
          out = thissearch;
        end
      otherwise
        error('Unrecognized logical search term.')
    end
  end
end

%-----------------------
function out = searchterm(sdata, myterm)
%Search for numeric data.
out = [];
%Don't run a straight eval command because unsafe.
rloc = find(ismember(str2cell(myterm(:)),{'<' '>' '=' '~'}));
if ~isempty(rloc) & rloc(1) == 1
  %This is the proper format so continue.
  termtype = myterm(rloc);
  myterm(rloc)=[];
  myterm = str2num(myterm);
  if length(myterm)~=1
    return
  end
  switch termtype
    case '<'
      out = sdata<myterm;
    case '>'
      out = sdata>myterm;
    case {'==' '='}
      out = sdata==myterm;
    case '>='
      out = sdata>=myterm;
    case '<='
      out = sdata<=myterm;
    case {'~=' '<>'}
      out = sdata~=myterm;      
  end
else
  myterm = str2num(myterm);
  out = ismember(sdata,myterm);
end

%-----------------------
function test

load arch
out = search(arch,'classid',1,1,'a');

a = dataset([1:100]);
a.axisscale{2} = [1:100];

out = search(a,'axisscale',2,1,'1:10and 20:30');
