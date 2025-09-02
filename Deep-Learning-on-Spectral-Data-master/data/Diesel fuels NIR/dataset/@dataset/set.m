function set(varargin)
%DATASET/SET Set field (property) values in DataSet data object.
%I/O: set(x)
%  displays all field names and possible values for the
%  input (x) which is a DataSet object (see DATASET).
%
%I/O: set(x,'field')  
%  displays the possible values for the input ('field').
%
%I/O: set(x,'field',value)
%  sets the field for a dataset object to (value).
%  This syntax is used for the following fields:
%    name         : char vector or cell of size 1 with char contents.
%    author       : char vector or cell of size 1 with char contents.
%    date         : 6 element vector (see CLOCK).
%    type         : field 'data' must not be empty to set 'type',
%                   can have value 'data' or 'image'.
%                   See documentation for type 'batch'.
%    data         : double array, or cell array (type 'batch' see documentation).
%    description  : char array or cell with char contents.
%    userdata     : filled with user defined data.
%
%I/O: set(x,'field',value,vdim)
%    include      : row vector of max length size(x.data,vdim) e.g. 1:size(x.data,vdim)
%I/O: set(x,'field',value,vdim,vset)
%  sets the field for a dataset object to (value) for the specified
%  dimension (vdim) and optional set (vset) {default: vset=1}, e.g.
%  the (vset) input allows for multiple sets of labels to be used.
%  The field 'data' must not be empty to set the values.
%  This syntax is used for the following fields:
%    label        : character or cell array with size(x.data,vdim)
%                   rows containing labels for the vdim dimension/mode.
%    labelname    : row vector of class char or 1 element cell containing
%                   a name for the label set.
%    axisscale    : vector of class double (real elements) with
%                   size(x.data,vdim) elements. Each contains an axis
%                   scale to plot against for the vdim dimension/mode
%                   (e.g. x.axisscale{2,1} is a numerical scale for columns,
%                   and x.axisscale{1,1} is a numerical scale for rows).
%                   See documentation for type 'batch'.
%    axistype     : row vector of class char with value of 'discrete'
%                   'stick' 'continuous' 'none'.
%    axisscalename: row vector of class char or 1 element cell containing 
%                   a name for the axis scale.
%    title        : row vector of class char or 1 element cell containing
%                   a title for the vdim dimension/mode.
%    titlename    : row vector of class char or 1 element cell containing
%                   a name for the title.
%    class        : vector of class double (integer elements) and size(x.data,vdim) 
%                   elements. Each element is a class identifier that can be used
%                   in plotting, classification, or crossvalidation.
%    classname    : row vector of class char or 1 element cell containing
%                   a name for the class set.
%    classlookup  : cell of nx2 where col 1 is double and col2 is string.
%                   Contains a lookup table relating class names to numeric
%                   class identifiers.
%    classid      : a reference that assigns/returns a cell array of stings
%                   based on the .classlookup table.
%
%See also: DATASET, DATASETDEMO, DATASET/GET, ISA, DATASET/SUBSASGN, DATASET/EXPLODE

%Copyright Eigenvector Research, Inc. 2000-2019
%nbg 8/3/00, 8/16/00, 8/17/00, 8/30/00, 9/1/00, 10/10/00
%nbg/jms 4/12/01 -fixed includ error if adding data to empty dataset
%jms 5/31/01 -revised to use subsasgn (centralized code)
%jms 5/21/02 - added test for not-a-real-dataset call (bug: should call builtin "get")
%jms 4/24/03 -renamed "includ" to "include"

%Fields 'label', 'axisscale', 'title', and 'class' can not be set if 'data'
%is empty. Once field 'data' has been filled it can not be "set" to a 
%different size.

%if x isn't a dataset, this was supposed to go to the built-in set function
if nargin > 0 & ~isa(varargin{1},'dataset');  
  builtin('set',varargin{:});
  return
end

%parse out varargin
if nargin >0;
  x = varargin{1};
end
if nargin >1;
  feyld = varargin{2};
end
if nargin >2;
  val = varargin{3};
end
if nargin >3;
  vdim = varargin{4};
end
if nargin >4;
  vset = varargin{5};
end

if nargin>1
  feyld = lower(feyld);
end

if nargin==1
  disp('            name:')
  disp('            type: [ {''data''} | ''image'' | ''batch'' ]')
  disp('          author:')
  disp('            date:')
  disp('         moddate:')
  disp('            data:')
  disp('           label:')
  disp('       axisscale:')
  disp('        axistype: [ {''discrete''} | ''stick'' | ''continuous''  | ''none'']')
  disp('           title:')
  disp('           class:')
  disp('     description:')
  disp('         history:')
  disp('         include:')
  disp('        userdata:')
elseif nargin==2
  if ~strcmp(class(feyld),'char')
      error('  Not valid dataset field.')
  end
  switch feyld
  case 'name'
    disp('  A dataset object''s name is a row vector of class char.')
  case 'author'
    disp('  A dataset object''s author is a row vector of class char.')
  case 'date'
    disp('  A dataset object''s date is a 6 element row vector (see CLOCK).')
  case 'moddate'
    disp('  A dataset object''s last modified date is a 6 element row vector.')
  case 'type'
    disp('  [ {''data''} | ''image'' | ''batch'']')
  case 'data'
    disp('  A dataset object''s data is an array of any numerical class ')
    disp('  or a column cell vector.')
  case 'label'
    disp('  A dataset object''s label is a cell array with contents')
    disp('  that are matrices of class char or cell. Rows of the cell')
    disp('  correspond to each mode/dimension of the ''data'' field.')
    disp('  Columns of the cell correspond to different label sets.')
  case 'labelname'
    disp('  A dataset object''s labelname is the name of a label set.')
  case 'axisscale'
    disp('  A dataset object''s axisscale is a cell array with contents')
    disp('  that are vectors of numerical scales of class double. Rows of')
    disp('  the cell correspond to each mode/dimension of the ''data'' field.')
    disp('  Columns of the cell correspond to different axisscale sets.')
  case 'axistype'
    disp('  A dataset object''s axistype is the name of the type of axis that')
    disp('  should be used to plot the ''axisscale''. The values for this')
    disp('  field can be ''discrete'', ''stick'', ''continuous'', or ''none''.')
    disp('  Columns of the cell correspond to different axisscale sets.')
  case 'axisscalename'
    disp('  A dataset object''s axisscalename is a name of a numerical scale set.')
  case 'title'
    disp('  A dataset object''s title is a cell array with contents')
    disp('  that are vectors of class char. Rows of the cell')
    disp('  correspond to each mode/dimension of the ''data'' field.')
    disp('  Columns of the cell correspond to different title sets.')
  case 'titlename'
    disp('  A dataset object''s titlename is a name of a title set.')
  case 'class'
    disp('  A dataset object''s class is a cell array with contents')
    disp('  that are vectors of class double. Rows of the cell')
    disp('  correspond to each mode/dimension of the ''data'' field.')
    disp('  Columns of the cell correspond to different class sets.')
  case 'classname'
    disp('  A dataset object''s classname is a name of a class set.')
  case 'classlookup'
    disp('  A dataset object''s classlookup table is a map for referencing')
    disp('  and assigning string values for classes. Values can be referenced')
    disp('  through the .classid field.')
  case 'classid'
    disp('  A dataset object''s classid field will return/assign a cell array of')
    disp('  strings relating numeric class values to their string identifiers')
    disp('  found in the classlookup table.')  
  case 'description'
    disp('  A dataset objects''s description is a matrix of class char')
    disp('  or cell.')
  case 'history'
    disp('  History of modifications/SET commands for dataset object.')
  case {'includ','include'}
    disp('  Cell array. Each cell contains a vector of indices.')
  case 'datasetversion'
    disp('  Version of a dataset object.')
  otherwise
    disp('  Not valid dataset object field.')
  end
elseif nargin>2
  
  switch nargin
  case 3
    indxstr = substruct('.',feyld);
  case 4
    indxstr = substruct('.',feyld,'{}',{vdim});
  case 5
    indxstr = substruct('.',feyld,'{}',{vdim vset});
  otherwise
    error('SET requires 3 to 5 inputs.')  
  end;
  
  try
    x = subsasgn(x,indxstr,val);
  catch
    errtxt = lasterr;
    
    try;
      while errtxt(end) == 10; errtxt(end)=[]; end;   %dump trailing line feeds
    catch;  end;
    
    if ~isempty(findstr(lasterr,'Index must be specified'))
      error([errtxt,10,'  Must use vdim and/or vset']);
    elseif ~isempty(findstr(lasterr,'Indicies not allowed'))
      error([errtxt,10,'  Do not use vdim and/or vset']);
    else
      error(errtxt);
    end;
  end;
  
  assignin('caller',inputname(1),x)
  
end;
