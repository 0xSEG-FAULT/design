function x=subsasgn(x,index,val)
%DATASET/SUBSASGN Assign for DataSet objects using Structure and index notation.
%I/O: x.field = value;
%  sets the field for a dataset object to (value).
%  This syntax is used for the following fields:
%    name         : char vector or cell of size 1 with char contents.
%    author       : char vector or cell of size 1 with char contents.
%    date         : 6 element vector (see CLOCK).
%    type         : field 'data' must not be empty to set 'type',
%                   can have value 'data' or 'image'.
%                   See documentation for type 'batch'.
%    imagesize    : vector size of image modes for type 'image' data.
%                   Should be used to unfold/refold image data in .data field.
%    imagemode    : mode where spatial data has been unfolded to.
%    image        : reference for image data, holds no data but will call
%                   function to display folded data.
%    data         : double, single, int or uint array, or cell array (type 'batch' see documentation).
%    description  : char array or cell with char contents.
%    userdata     : filled with user defined data.
%    history      : appends string or cell array of strings as comments to
%                   history field. Comments are always preceeded by %%%
%
%I/O: x.field{vdim} = value;
%    include      : row vector of max length size(x.data,vdim) e.g. 1:size(x.data,vdim)
%
%I/O: x.field{vdim,vset} = value;
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
%    imageaxisscale : vector of class double (real elements) with
%                   length x.imagesize elements. Each contains an axis
%                   scale to plot against for the image mode.
%                   Not available for type 'batch'.
%    axistype     : row vector of class char with value of 'discrete'
%                   'stick' 'continuous' 'none'.
%    imageaxistype : row vector of class char with value of 'discrete'
%                   'stick' 'continuous' 'none'.
%    axisscalename: row vector of class char or 1 element cell containing
%                   a name for the axis scale.
%    imageaxisscalename : row vector of class char or 1 element cell containing
%                   a name for the image axis scale.
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
%Examples:
%    mydataset.author = 'James Joyce'
%    mydataset.axisscale{2} = [1:.5:25]        %second dim axis scale
%    mydataset.title{1,2} = 'Second title for first dim'
%
%See also: DATASET, DATASETDEMO, DATASET/SUBSREF, DATASET/GET, ISA, DATASET/EXPLODE

%Copyright Eigenvector Research, Inc. 2000-2019
%nbg 8/3/00, 8/16/00, 8/17/00, 8/30/00, 9/1/00, 10/10/00
%jms 4/11/01
%   -this started off as nearly the same as SET for datasets except
%     that we added the logic to take input like a subsasgn call and
%     translate it into the correct info for the old set routine
%   -fixed error in errors (missing [ ]s)
%   -modified error statements to make more sense with this kind of call
%   -changed history logging function (renamed old one to keep it)
%nbg/jms 4/12/01 -fixed includ error if adding data to empty dataset
% jms 5/9/01 -added handling of other classes for data field
% jms 5/31/01 -added subscripting into fields for assignment
%   -fixed bug which kept batch data axisscales from be addressed correctly
%   -fixed bugs in data assign checking batch data size
% jms 6/5/01 -fixed includ size bug (adding data to empty dataset created
%    includ field with only 1 column)
% jms 10/19/01 -added error if vset used on includ field
%   -added error if subscripting used on author field
%   -added calling function identification in sethistory
% jms 11/05/01 -axisscale and label can now be set to empty []
% jms 4/16/02  -removed extraneous code in sethistory
% jms 5/20/02 -changed calls to sethistory and sethistory output format
% jms 8/02 -removed extraneous tests of data class
%    -revised error messages
%    -fixed bugs associated with batch support
% jms 9/02 -changed subscripting error message
% jms 4/24/03 -renamed "includ" to "include"
% jms 11/7/03 -base vdim tests on ndims(x) code instead of ndims(x.data)
% rsk 02/16/05 -fix bug in imagesize checking, ndims < 2.

%Fields 'label', 'axisscale', 'title', and 'class' can not be set if 'data'
%is empty. Once field 'data' has been filled it can not be "set" to a
%different size.

nnargin = nargin;
error(nargchk(3,3,nnargin));   %give error if nargin is not appropriate
myindex = index;
if isempty(index); error('  Invalid subscripting'); end;

if ~isa(x,'dataset')
  %val is Dataset, not X. Don't processes this here
  x = builtin('subsasgn',x,index,val);
  return
end

if strcmp(index(1).type,'()') & isempty(val)
  %  x(_,_) = []    %delete rows/cols/etc
  % translate to  x.data(_,_) = []  
  index = [substruct('.','data');index];
elseif ~strcmp(index(1).type,'.');
  error('  Assignment subscripting not permitted on main dataset object');
end
feyld=lower(index(1).subs);

if length(index)<2 | ~strcmp(index(2).type,'{}') | ...
    (strcmp(x.type,'batch') & strcmp(feyld,'data'));
  %Make set and dim equal to one so retrieve entire field in nnargin=2 case.
  %Calls without vdim or vset (e.g. on data, date, author, etc)
  vdim  = 1;
  vset  = 1;
  nnargin = 3;     %fake out logic later on
  index(1) = [];    %clear field name from index structure
else
  %Indexing into specific location.
  vdim=index(2).subs{1};
  if length(index(2).subs)>1;
    vset=index(2).subs{2};
    nnargin = 5;     %fake out logic later on
  else
    vset=1;
    nnargin = 4;     %fake out logic later on
  end;
  index(1:2) = [];  %clear field name & vset/vdim from index structure
  %Remaining parts of index will be used to subscript into results after
  %getting results.
end;

if (length(vdim)>1 & (~isa(x.data,'cell') | ~strcmp(feyld,'axisscale'))) | ...
    (length(vdim)>2 & isa(x.data,'cell') & strcmp(feyld,'axisscale')) | length(vset)>2;
  error('  Indicies must be single integer values');
end;

ts = [];  %to hold timestamp (when/if history entry is made)
if nnargin==3
  if ~strcmp(class(feyld),'char')
    error('  Not valid dataset object field.')
  end
  switch feyld
    case 'name'
      if ~isempty(index);
        error(['  Field ''' feyld ''' does not allow assignment subscripting']);
      end;
      if ~(strcmp(class(val),'char')|strcmp(class(val),'cell'))
        error('  Value must be class char or cell for field ''name''.')
      elseif size(val,1)>1|ndims(val)>2
        error('  Size(value,1) & ndims(value) must = 1 for field ''name''.')
      elseif strcmp(class(val),'cell')&size(val,2)>1
        error('  Cell input must be size 1x1 for field ''name''.')
      else
        if strcmp(class(val),'cell')
          val    = char(val{:});
        end
        x.name    = val;
        [x.history,ts] = sethistory(x.history,'','name',['''' val '''']);
      end
    case 'author'
      if ~isempty(index);
        error(['  Field ''' feyld ''' does not allow assignment subscripting']);
      end;
      if ~(strcmp(class(val),'char')|strcmp(class(val),'cell'))
        error('  Value must be class char or cell for field ''author''.')
      elseif (size(val,1)>1)|ndims(val)>2
        error('  Size(value,1) & ndims(value) must = 1 for field ''author''.')
      elseif strcmp(class(val),'cell')&size(val,2)>1
        error('  Cell input must be size 1x1 for field ''author''.')
      else
        if strcmp(class(val),'cell')
          val    = char(val{:});
        end
        x.author = val;
        [x.history,ts] = sethistory(x.history,'','author',['''' val '''']);
      end
    case 'date'
      if isempty(x.date)
        if ~isempty(index);
          error(['  Field ''' feyld ''' does not allow assignment subscripting']);
        end;
        if ~strcmp(class(val),'double')
          error('  Value must be class double for field ''date''.')
        elseif (size(val,1)>1)|(size(val,2)~=6)|(ndims(val)>2)
          error('  Value must be a 6 element row vector for field ''date''.')
        else
          x.date    = val;
          x.moddate = val;
        end
      else
        error('  Field ''date'' can''t be reset.')
      end
    case 'moddate'
      error('  Field ''moddate'' can''t be reset.')
    case 'type'
      if ~isempty(index);
        error(['  Field ''' feyld ''' does not allow assignment subscripting']);
      end;
      if ~(strcmp(class(val),'char')|strcmp(class(val),'cell'))
        error('  Invalid data object field value.')
      elseif ~(strcmp(val,'data')| ...
          strcmp(val,'image')| ...
          strcmp(val,'batch'))
        error('  Invalid data structure field value.')
      elseif isempty(x.data)
        error(['  ''' feyld ''' can not be set when ''data'' is empty.'])
      elseif strcmp(val,'image')
        if ~strcmp(x.type, val)
          %Changing DSO type, check image size/mode fields.
          %Set to default values if empty.
          if isempty(x.imagesize)
            x.imagesize = [size(x.data,1) 1];
          end
          if isempty(x.imagemode)
            x.imagemode = 1;
          end
        end
        x.type = val;
        [x.history,ts] = sethistory(x.history,'','type',['''' val '''']);
      elseif strcmp(val,'batch')
        if ~isa(x.data,'cell')
          error('  class(x.data) must be cell for type batch.')
        else
          x.type = batch;
        end
      else
        x.type   = val;
        [x.history,ts] = sethistory(x.history,'','type',['''' val '''']);
      end
    case 'imagesize'
      if ~strcmp(x.type,'image')
        error(' Dataset type must be ''image'' to assign a value to this field.')
      end
      if ~isempty(index);
        error(['  Field ''' feyld ''' does not allow assignment subscripting']);
      end;
      if ~strcmp(class(val),'double')
        error('  Value must be class double for field ''imagesize''.')
      elseif (prod(size(val))~=size(val,2))|(size(val,2)< 2)
        error(['  Value must be a row vector with a length less than the',...
          ' total number of data modes for field ''imagesize''.'])
      elseif prod(val) ~= size(x.data,x.imagemode)
        error(['Image size does not match number of available pixels (mode ' num2str(x.imagemode) '). Please check data size.']);
      else
        x.imagesize    = val;
        %Reset image axisscale if changing image dims.
        x.imageaxisscale = cell(length(val),2,1);
        x.imageaxistype = repmat({'none'},length(val),1);
        [x.history,ts] = sethistory(x.history,'','imagesize',['''' num2str(val) '''']);
      end
    case 'imagemode'
      if ~strcmp(x.type,'image')
        error(' Dataset type must be ''image'' to assign a value to this field.')
      end
      if ~isempty(index);
        error(['  Field ''' feyld ''' does not allow assignment subscripting']);
      end;
      if ~strcmp(class(val),'double')
        error('  Value must be class double for field ''imagemode''.')
      elseif (sum(size(val))~=2)|(val>ndims(x))
        error(['  Value must be a scalar with a value less than the',...
          ' total number of data modes for field ''imagemode''.'])
      else
        x.imagemode    = val;
        [x.history,ts] = sethistory(x.history,'','imagemode',['''' num2str(val) '''']);
      end
    case 'imagemap'
      if ~strcmp(x.type,'image')
        error(' Dataset type must be ''image'' to use this field.')
      end
      error('  Field ''imagemap'' can''t be reset.')
    case 'imagedata'
      if ~strcmp(x.type,'image')
        error(' Dataset type must be ''image'' to use this field.')
      end
      error('  Field ''imagedata'' can''t be reset.')
    case 'data'
      if ~any(strcmp(class(val),{'double','single','logical','int8','int16','int32','uint8','uint16','uint32','cell'}))
        error('  Field ''data'' must be of class double, single, logical, [u]int8/16/32, or cell.')
      end;
      if ~isempty(x.data)
        if isempty(val)
          %DELETE assignment!
          indmode = find(cellfun(@(i) isnumeric(i)&~isempty(i),index.subs,'uniformoutput',true));
          if length(indmode)~=1
            error('Subscripted assignment dimension mismatch.')
          end
          x = delsamps(x,index.subs{indmode},indmode,2); %hard delete
        elseif ~isa(val,'cell')
          if strcmp(x.type,'batch') | isa(x.data,'cell')
            error('  Value must be class cell for dataset of type ''batch''.')
          end
          if ~isempty(index);
            try
              x.data = subsasgn(x.data,index,val);
            catch
              error(lasterr)
            end
          else
            %get sizes of current and new data
            szval = size(val);
            szx   = size(x);
            %make size vectors match in length (pad with 1's)
            szval(end+1:length(szx)) = 1;
            if ~all(size(x.data)==size(val))
              error('  Size of value must equal size of previously loaded data (see dataset/cat and dataset/delsamps).')
            end
            x.data = val;
          end
          if ~iscell(x.include)
            x.include = cell(ndims(val),2);
          end
          for ii=1:ndims(val)
            if isempty(x.include{ii});  %any empty cell resets all cells
              for iii=1:ndims(val)
                x.include{iii} = 1:size(val,iii);
              end;
              break;    %no point in looking further, we just reset them all
            end;
          end;
          notes  = int2str(size(val,1));
          for ii=2:ndims(val)
            notes = [notes,'x',int2str(size(val,ii))];
          end
          if isempty(index);
            notes   = ['class ' class(val) ' ',notes,' entire data array replaced.'];
          else
            notes   = ['class ' class(val) ' ',notes,' partial data array replaced.'];
          end;
          [x.history,ts] = sethistory(x.history,'','data','',[],[],notes);
        else    %batch mode
          if ~strcmp(x.type,'batch') | ~isa(x.data,'cell')
            error('  Value cannot be class cell for dataset of type ''data'' or ''image''.')
          end
          if (size(val,1)>1) & (size(val,2)>1)
            error('Multidimensional cells not valid for dataset of type ''batch''.')
          end
          if ~isempty(index);
            try
              x.data = subsasgn(x.data,index,val);
            catch
              error(lasterr)
            end
          else
            csize   = size(val{1});
            csize   = csize(2:end);    %size of dimensions~=1
            for ii=2:length(val)       %make certain that contents of all cells are same size except dim 1
              csize2 = size(val{ii});
              csize2 = csize2(2:end);
              if any(csize2~=csize)
                error('All modes except 1 must be same size.')
              end
            end
            x.data  = val;
          end
          notes   = int2str(size(val,1));
          for ii=2:ndims(val{1})
            notes = [notes,'x',int2str(size(val,ii))];
          end
          if isempty(index);
            notes   = ['class cell ',notes,' entire cell array replaced.'];
          else
            notes   = ['class cell ',notes,' partial cell array replaced.'];
          end;
          [x.history,ts] = sethistory(x.history,'','data','',[],[],notes);
          
        end
      else
        if ~isempty(index);
          error(['  Field ''Data'' is empty and cannot allow assignment subscripting']);
        end;
        temp             = dataset(val);  %use basic dataset function to create the appropriate dataset
        temp.name        = x.name;
        temp.author      = x.author;
        temp.description = x.description;
        temp.userdata    = x.userdata;
        temp.history     = x.history;
        x = temp;
        
        if ~isa(val,'cell')
          nmodes = ndims(val);
        else
          nmodes = ndims(val{1});
        end
        notes    = int2str(size(val,1));
        for ii=2:nmodes
          notes  = [notes,'x',int2str(size(val,ii))];
        end
        notes    = ['class ',class(val),' ',notes];
        [x.history,ts] = sethistory(x.history,'','data','',[],[],notes);
      end
    case 'label'
      error('  Index must be specified for field ''label''.')
    case 'labelname'
      error('  Index must be specified for field ''labelname''.')
    case 'axisscale'
      error('  Index must be specified for field ''axisscale''.')
    case 'axisscalename'
      error('  Index must be specified for field ''axisscalename''.')
    case 'axistype'
      error('  Index must be specified for field ''axistype''.')
    case 'imageaxisscale'
      if ~strcmp(x.type,'image')
        error('  Field ''imageaxisscale'' cannot be set when type is not ''image''.')
      end
      if ~iscell(val) | length(val)~=length(x.imagesize)
        error('  Index must be specified for field ''imageaxisscale'' or cell array of appropriate length must be supplied.')
      end
      for j=1:length(val)
        if (size(val{j},1)>1&size(val{j},2)>1)|(ndims(val{j})>2)|(~isa(val{j},'double'))
          error('  Value must be vector of class double.')
        elseif x.imagesize(j)~=length(val{j}) & ~isempty(val{j});
          error(['  Length of input ''imageaxisscale'' must equal ', ...
            'size of image (x.imagesize(vdim)) = ',int2str(x.imagesize(j)),'.'])
        end
        x.imageaxisscale{j,1,1} = val{j};
      end
      [x.history,ts] = sethistory(x.history,'','imageaxisscale','',[],[],'All image axes changed');
    case 'imageaxisscalename'
      if ~strcmp(x.type,'image')
        error('  Field ''imageaxisscalename'' cannot be set when type is not ''image''.')
      end
      if ~iscell(val) | length(val)~=length(x.imagesize)
        error('  Index must be specified for field ''imageaxisscalename'' or cell array of appropriate length must be supplied.')
      end
      for j=1:length(val)
        if ~isa(val{j},'char') & ~isempty(val{j})
          error('  Value must be a string.')
        end
        x.imageaxisscale{j,2,1} = char(val{j});
      end
      [x.history,ts] = sethistory(x.history,'','imageaxisscalename','',[],[],'All image axes changed');
    case 'title'
      error('  Index must be specified for field ''title''.')
    case 'titlename'
      error('  Index must be specified for field ''titlename''.')
    case 'class'
      error('  Index must be specified for field ''class''.')
    case 'classid'
      error('  Index must be specified for field ''classid''.')
    case 'classname'
      error('  Index must be specified for field ''classname''.')
    case 'classlookup'
      error('  Index must be specified for field ''classlookup''.')
    case 'description'
      if ~isempty(index);
        error(['  Field ''' feyld ''' does not allow assignment subscripting']);
      end;
      if ~(strcmp(class(val),'char')|strcmp(class(val),'cell'))
        error('  Field ''description'' must be of class char or cell.')
      else
        if strcmp(class(val),'cell')
          val         = char(val{:});
        end
        x.description = val;
        [x.history,ts] = sethistory(x.history,'','description','');
      end
    case {'includ','include'}
      error('  Index must be specified for field ''include''.')
    case 'datasetversion'
      error('  Field ''datasetversion'' can not be modified.')
    case 'uniqueid'
      error('  Field ''uniqueid'' can not be modified.')
    case 'userdata'
      if isempty(index);
        x.userdata = val;
      else
        try
          x.userdata = subsasgn(x.userdata,index,val);
          val = x.userdata;
        catch; error(lasterr); end;
      end;
      notes     = [int2str(size(val,1))];
      for ii=2:ndims(val)
        notes   = [notes,'x',int2str(size(val,ii))];
      end
      notes     = ['class ',class(val),' ',notes];
      if ~isempty(index);
        notes = [notes ' (subscript replacement)'];
      end;
      [x.history,ts] = sethistory(x.history,'','userdata','',[],[],notes);
      
    case 'history'
      %add item(s) passed (as string, string array, or cell array of strings)
      %to history field as COMMENTS
      cval = val;
      if ~iscell(cval)
        if ischar(cval) & size(cval,1)>1;
          cval = str2cell(cval);
        else
          cval = {cval};
        end
      end
      for j=1:length(cval);
        val = cval{j};
        if ~ischar(val) | size(val,1)>1
          error('Assignment of comment into history can only accept single-line string');
        end
        val(val<32) = [];  %drop ALL control characters
        [tstamp,ts] = timestamp;
        x.history{end+1} = sprintf('%%%%%% Comment: %s (%s)',val,tstamp);
      end
      
    otherwise
      
      [value,match] = findlabelmatch(x,myindex(1).subs);
      
      %if ~isempty(match.subs(2))
         % This is the case where the user indexes into the data array
         % Example: wine.Beer.data(1:5) = [1:5] 
 
       if ~isempty(match)  
         if length(myindex) > 2 
            if ~isempty(myindex(3)) & strcmpi(myindex(3).type,'()')
     
            idx = match.subs(2);
            idx = idx{1};
            myindex(3).subs{2} = idx;
            x.data = subsasgn(x.data,myindex(3),val);
            end 
         else
          
          % Case where user provides no indices, assume that the assigned value is the length of a column 
          % Example: wine.Beer.data = 
          idx  = match.subs(2);
          idx = idx{1};
          myindex(3).type ='()';
          myindex(3).subs{2} = idx;
          myindex(3).subs{1} = [1:size(x.data,1)];
          x.data = subsasgn(x.data,myindex(3),val);
         end
       else
         error('  "%s" is not a valid variable label or "%s" is not a valid dataset object field. ',myindex(1).subs, myindex(1).subs ) 
      end
      
      %error('  "%s" is not a valid dataset object field.',feyld)
      
  end
  if isempty(ts)
    ts = clock;
  end
  if isempty(x.date)
    x.date = ts;
  end
  x.moddate = ts;
  
elseif nnargin==4|nnargin==5
  if ~strcmp(class(feyld),'char')
    error('  Not valid DataSet field.')
  elseif (~isa(x.data,'cell')&(length(vdim)>1))
    error('  vdim must be scalar.')
  elseif ismember(feyld,{'imageaxisscale' 'imageaxisscalename' 'imageaxistype'})
    %It's possible image axis scale index could exceed the size of the
    %unfolded data so check here.
    if vdim>length(x.imagesize)
      error('  vdim cannot be > length(x.imagesize).')
    end
  elseif ~isa(x.data,'cell') & vdim>ndims(x)
    error('  vdim cannot be > ndims(x).')
  elseif isa(x.data,'cell') & vdim>ndims(x.data{1})
    error('  vdim cannot be > ndims(x.data{1}).')
  end
  switch feyld
    case 'name'
      error('  Indicies not allowed for field ''name''.')
    case 'type'
      error('  Indicies not allowed for field ''type''.')
    case 'author'
      error('  Indicies not allowed for field ''author''.')
    case 'data'
      error('  Indicies not allowed for field ''data''.')
    case 'history'
      error('  Indicies not allowed for field ''history''.')
    case 'label'
      if ~isempty(index);
        if vset>size(x.label,3);
          error(['  ''label'' set ' num2str(vset) ' does not exist.'])
        end
        if strcmp(index(1).type,'()') & (length(index(1).subs)<2 | (~isnumeric(index(1).subs{2}) & strcmpi(index(1).subs{2},':')))
          %convert:  x.label{__}(rows,:) = {'string' 'string'} into
          %  x.label{__}{rows} = {'string' 'string'}
          index(1).type = '{}';
          index(1).subs = index(1).subs(1);  %drop all but row index
        end            
        if strcmp(index(1).type,'()')
          try
            val = subsasgn(x.label{vdim,1,vset},index,val);
          catch; error(lasterr); end;
        elseif strcmp(index(1).type,'{}')
          inds = index(1).subs{1};
          if ~iscell(val)
            val = {val};
          end
          allval = x.label{vdim,1,vset};
          allval = str2cell(allval);
          allval(inds) = val;
          val = char(allval);
        end
      end;
      if isempty(x.data)
        error('  ''label'' can not be set when ''data'' is empty.')
      elseif ~(strcmp(class(val),'char')|strcmp(class(val),'cell'))
        error('  ''label'' Value must be array of class char or cell.')
      elseif strcmp(class(val),'cell')
        notes     = ['class cell ', int2str(size(val,1))];
        for ii=2:ndims(val)
          notes     = [notes,'x',int2str(size(val,ii))];
        end
        val   = char(val{:});
      else
        notes = ['class char ', int2str(size(val,1))];
        for ii=2:ndims(val)
          notes     = [notes,'x',int2str(size(val,ii))];
        end
      end
      if ~isa(x.data,'cell') | vdim == 1
        if size(x.data,vdim)~=size(val,1) & ~isempty(val);
          error(['  Number of rows (or cell elements) in input ''label'' must =', ...
            'size(x.data,vdim) = ',int2str(size(x.data,vdim)),'.'])
        end
      else
        if size(x.data{1},vdim)~=size(val,1) & ~isempty(val);
          error(['  Number of rows (or cell elements) in input ''label'' must =', ...
            'size(x.data{1},vdim) = ',int2str(size(x.data{1},vdim)),' (for vdim > 1).'])
        end
      end
      
      nsets = size(x.label,3);
      if nsets<vset;
        [x.label{:,:,nsets+1:vset}] = deal('');   %expand to new set(s) with empty STRINGS
      end
      x.label{vdim,1,vset} = val;
      if ~isempty(index);
        notes = [notes ' (subscript replacement)'];
      end;
      [x.history,ts] = sethistory(x.history,'','label','',vdim,vset,notes);
      
    case 'labelname'
      if ~isempty(index);
        error(['  Field ''' feyld ''' does not allow assignment subscripting']);
      end;
      if isempty(x.data)
        error(['  ''' feyld ''' can not be set when ''data'' is empty.'])
      elseif isempty(val)
        val = char(val);
      elseif ~(strcmp(class(val),'char')|strcmp(class(val),'cell'))
        error('  Value must be class char or cell for field ''labelname''.')
      elseif strcmp(class(val),'cell')
        val    = char(val{:});
      end
      if size(val,1)>1
        error(['  Size(value,1) (or number of cell elements) must ', ...
          '= 1 for field ''labelname''.'])
      end
      nsets = size(x.label,3);
      if nsets<vset;
        [x.label{:,:,nsets+1:vset}] = deal('');   %expand to new set(s) with empty STRINGS
      end
      x.label{vdim,2,vset} = val;
      [x.history,ts] = sethistory(x.history,'','labelname',['''' val ''''],vdim,vset);
      
    case 'axisscale'
      if isempty(x.data)
        error(['  ''' feyld ''' can not be set when ''data'' is empty.'])
      end;
      if isa(x.data,'cell') & ~isempty(vdim) & vdim(1) == 1 | length(vdim)>1;  %type batch, first mode assignment
        if isa(val,'cell') & length(vdim)==1 & vdim==1;
          %assignment of all first-mode axisscales at once through a cell
          if ~isempty(index);
            if vset>size(x.axisscale,3);
              error(['  ''axisscale'' set ' num2str(vset) ' does not exist.'])
            end
            try
              val = subsasgn(x.axisscale{1,1,vset},index,val);
            catch; error(lasterr); end;
          end;
          if ~all(size(x.data)==size(val))  & ~isempty(val);% val must be cell of length(x.data)
            error(['  Cell size must = length(x.data) = ' int2str(length(x.data))])
          end
          for ii=1:length(val)
            if all(size(val{ii})>1) | ndims(val{ii})>2
              error(['Cell contents (value{',int2str(ii),'}) must be vector.'])
            end
            if length(val{ii})~=size(x.data{ii},1)
              error(['length(value{',int2str(ii),'}) must = ', ...
                'size(x.data{',int2str(ii),'},1).'])
            end
            if size(val{ii},1)>size(val{ii},2)
              val{ii}  = val{ii}';
            end
          end
          
          nsets = size(x.axisscale,3);
          if nsets<vset;
            [x.axisscale{:,2,nsets+1:vset}] = deal('');   %expand to new set(s) with empty STRINGS
          end
          x.axisscale{1,1,vset} = val;
          
          notes = ['class cell ', int2str(size(val,1))];
          for ii=2:ndims(val)
            notes   = [notes,'x',int2str(size(val,ii))];
          end
          if ~isempty(index);
            notes = [notes ' (subscript replacement)'];
          end;
          [x.history,ts] = sethistory(x.history,'','axisscale','',vdim,vset,notes);
        else  %vdim must be 2 element vector w/ vdim(1)==1
          if ~isempty(index);
            error(['  Multi-indexed batch axisscale does not allow assignment subscripting. Use scalar vdim.']);
          end;
          if (prod(size(vdim))>2)|(prod(size(vdim))<2)
            error('vdim must be 2 element vector with vdim(1)==1.')
          elseif vdim(2)>length(x.data)
            error(['vdim(2)>number of cells ',int2str(length(x.data))])
          end
          if ~isa(val,'double')
            error('value must be vector of class double.')
          elseif (size(val,1)>1&size(val,2)>1)|ndims(val)>2
            error('value must be vector of class double.')
          elseif length(val)~=size(x.data{vdim(2)},1)
            error('length(value)~=size(x.data{vdim(2)},1)')
          elseif size(val,1)>size(val,2)
            notes   = ['class double ',int2str(size(val,1)),'x1'];
            val     = val';
          else
            notes   = ['class double 1x',int2str(size(val,1))];
          end
          
          nsets = size(x.axisscale,3);
          if nsets < vset;
            [x.axisscale{1,1,nsets+1:vset}] = deal(cell(length(x.data),1));  %create new empty set if set didn't exist yet
            [x.axisscale{:,2,nsets+1:vset}] = deal('');   %expand to new set(s) with empty STRINGS
            %Add place holders for axistype.
            [x.axistype{:,nsets+1:vset}] = deal('none');
          end
          
          %Add an axis type if none present.
          if isempty(x.axistype{vdim,vset})
            x.axistype{vdim,vset} = 'none';
          end
          
          x.axisscale{1,1,vset}{vdim(2)} = val;
          if ~isempty(index);
            notes = [notes ' (subscript replacement)'];
          end;
          [x.history,ts] = sethistory(x.history,'','axisscale','',vdim,vset,notes);
        end
        
      else %x.data is class 'double' or it's 'cell' w/ vdim>1
        if ~isempty(index);
          if vset>size(x.axisscale,3);
            error(['  ''axisscale'' set ' num2str(vset) ' does not exist.'])
          end
          try
            val = subsasgn(x.axisscale{vdim,1,vset},index,val);
          catch; error(lasterr); end;
        end;
        if (size(val,1)>1&size(val,2)>1)|(ndims(val)>2)|(~isa(val,'double'))
          error('  Value must be vector of class double.')
        elseif ~isa(x.data,'cell') & size(x.data,vdim)~=length(val) & ~isempty(val);
          error(['  Length of input ''axisscale'' must = ', ...
            'size(x.data,vdim) = ',int2str(size(x.data,vdim)),'.'])
        elseif isa(x.data,'cell') & size(x.data{1},vdim)~=length(val) & ~isempty(val);
          error(['  Length of input ''axisscale'' must = ', ...
            'size(x.data{1},vdim) = ',int2str(size(x.data{1},vdim)),'.'])
        end
        
        if size(val,1)>size(val,2)
          val  = val';
        end
        notes = ['class double ', int2str(size(val,1))];
        for ii=2:ndims(val)
          notes   = [notes,'x',int2str(size(val,ii))];
        end
        
        nsets = size(x.axisscale,3);
        if nsets<vset;
          [x.axisscale{:,2,nsets+1:vset}] = deal('');   %expand to new set(s) with empty STRINGS
          %Add place holders for axistype.
          [x.axistype{:,nsets+1:vset}] = deal('none');
        end
        
        x.axisscale{vdim,1,vset} = val;
        
        %Add an axis type if none present.
        if isempty(x.axistype{vdim,vset})
          x.axistype{vdim,vset} = 'none';
        end
        
        if ~isempty(index);
          notes = [notes ' (subscript replacement)'];
        end;
        [x.history,ts] = sethistory(x.history,'','axisscale','',vdim,vset,notes);
        
      end
    case 'imageaxisscale'
      %Need to be type image.
      if ~strcmp(x.type,'image')
        error('DSO must be type "image" to assigning imageaxisscale.')
      end
      if isempty(x.data)
        error(['  ''' feyld ''' can not be set when ''data'' is empty.'])
      end;
      if isa(x.data,'cell')  %type batch, first mode assignment
        error('Cannot assign imageaxisscale with batch type DSO.')
      else
        if ~isempty(index);
          if vset>size(x.imageaxisscale,3);
            error(['  ''imageaxisscale'' set ' num2str(vset) ' does not exist.'])
          end
          try
            val = subsasgn(x.imageaxisscale{vdim,1,vset},index,val);
          catch; error(lasterr); end;
        end;
        if (size(val,1)>1&size(val,2)>1)|(ndims(val)>2)|(~isa(val,'double'))
          error('  Value must be vector of class double.')
        elseif ~isa(x.data,'cell') & x.imagesize(vdim)~=length(val) & ~isempty(val);
          error(['  Length of input ''imageaxisscale'' must equal ', ...
            'size of image (x.imagesize(vdim)) = ',int2str(x.imagesize(vdim)),'.'])
        end
        
        if size(val,1)>size(val,2)
          val  = val';
        end
        notes = ['class double ', int2str(size(val,1))];
        for ii=2:ndims(val)
          notes   = [notes,'x',int2str(size(val,ii))];
        end
        
        nsets = size(x.imageaxisscale,3);
        if nsets<vset;
          [x.imageaxisscale{:,2,nsets+1:vset}] = deal('');   %expand to new set(s) with empty STRINGS
          %Add place holders for axistype.
          [x.imageaxistype{:,nsets+1:vset}] = deal('none');
        end
        
        x.imageaxisscale{vdim,1,vset} = val;
        
        %Add an axis type if none present.
        if isempty(x.imageaxistype{vdim,vset})
          x.imageaxistype{vdim,vset} = 'none';
        end
        
        if ~isempty(index);
          notes = [notes ' (subscript replacement)'];
        end;
        x.history = sethistory(x.history,'','imageaxisscale','',vdim,vset,notes);
        
      end
    case 'axisscalename'
      if ~isempty(index);
        error(['  Field ''' feyld ''' does not allow assignment subscripting']);
      end;
      if isempty(x.data)
        error(['  ''' feyld ''' can not be set when ''data'' is empty.'])
      elseif isempty(val)
        val = char(val);
      elseif ~(strcmp(class(val),'char')|strcmp(class(val),'cell'))
        error('  Value must be class char or cell for field ''axisscalename''.')
      elseif strcmp(class(val),'cell')
        val    = char(val{:});
      end
      if size(val,1)>1
        error(['  Size(value,1) (or number of cell elements) must ', ...
          '= 1 for field ''axisscalename''.'])
      end
      
      nsets = size(x.axisscale,3);
      if nsets<vset;
        [x.axisscale{:,2,nsets+1:vset}] = deal('');   %expand to new set(s) with empty STRINGS
        %Add place holders for axistype.
        [x.axistype{:,nsets+1:vset}] = deal('none');
      end
      x.axisscale{vdim,2,vset}= val;
      
      [x.history,ts] = sethistory(x.history,'','axisscalename',['''' val ''''],vdim,vset);
    case 'imageaxisscalename'
      if ~isempty(index);
        error(['  Field ''' feyld ''' does not allow assignment subscripting']);
      end;
      if isempty(x.data)
        error(['  ''' feyld ''' can not be set when ''data'' is empty.'])
      elseif isempty(val)
        val = char(val);
      elseif ~(strcmp(class(val),'char')|strcmp(class(val),'cell'))
        error('  Value must be class char or cell for field ''imageaxisscalename''.')
      elseif strcmp(class(val),'cell')
        val    = char(val{:});
      end
      if size(val,1)>1
        error(['  Size(value,1) (or number of cell elements) must ', ...
          '= 1 for field ''imageaxisscalename''.'])
      end
      
      nsets = size(x.imageaxisscale,3);
      if nsets<vset;
        [x.imageaxisscale{:,2,nsets+1:vset}] = deal('');   %expand to new set(s) with empty STRINGS
        %Add place holders for axistype.
        [x.imageaxistype{:,nsets+1:vset}] = deal('none');
      end
      x.imageaxisscale{vdim,2,vset}= val;
      
      x.history = sethistory(x.history,'','imageaxisscalename',['''' val ''''],vdim,vset);
      
    case 'axistype'
      if ~isempty(index);
        error(['  Field ''' feyld ''' does not allow assignment subscripting']);
      end;
      if isempty(x.data)
        error(['  ''' feyld ''' can not be set when ''data'' is empty.'])
      elseif ~(strcmp(class(val),'char')|strcmp(class(val),'cell'))
        error('  Value must be class char or cell for field ''axistype''.')
      elseif strcmp(class(val),'cell')
        val    = char(val{:});
      end
      if size(val,1)>1
        error(['  Size(value,1) (or number of cell elements) must ', ...
          '= 1 for field ''axistype''.'])
      end
      if isempty(val)
        val = 'none';
      end
      if ~ismember(val,{'discrete' 'stick' 'continuous' 'none'})
        error(['  Axistype can only be: ''discrete'' ''stick'' ''continuous'' ''none''.'])
      end
      
      nsets = size(x.axistype,2);%?
      if nsets<vset;
        [x.axistype{:,nsets+1:vset}] = deal('none');   %expand to new set(s) with empty STRINGS
      end
      x.axistype{vdim,vset}= val;
      [x.history,ts] = sethistory(x.history,'','axistype',['''' val ''''],vdim,vset);
    case 'imageaxistype'
      if ~isempty(index);
        error(['  Field ''' feyld ''' does not allow assignment subscripting']);
      end;
      if isempty(x.data)
        error(['  ''' feyld ''' can not be set when ''data'' is empty.'])
      elseif ~(strcmp(class(val),'char')|strcmp(class(val),'cell'))
        error('  Value must be class char or cell for field ''imageaxistype''.')
      elseif strcmp(class(val),'cell')
        val    = char(val{:});
      end
      if size(val,1)>1
        error(['  Size(value,1) (or number of cell elements) must ', ...
          '= 1 for field ''imageaxistype''.'])
      end
      if ~ismember(val,{'discrete' 'stick' 'continuous' 'none'})
        error(['  Imageaxistype can only be: ''discrete'' ''stick'' ''continuous'' ''none''.'])
      end
      
      nsets = size(x.imageaxistype,2);%?
      if nsets<vset;
        [x.imageaxistype{:,nsets+1:vset}] = deal('none');   %expand to new set(s) with empty STRINGS
      end
      x.imageaxistype{vdim,vset}= val;
      x.history = sethistory(x.history,'','imageaxistype',['''' val ''''],vdim,vset);
      
    case 'title'
      if ~isempty(index);
        error(['  Field ''' feyld ''' does not allow assignment subscripting']);
      end;
      if isempty(x.data)
        error(['  ''' feyld ''' can not be set when ''data'' is empty.'])
      elseif ~(strcmp(class(val),'char')|strcmp(class(val),'cell'))
        error('  Value must be class char or cell for field ''title''.')
      elseif strcmp(class(val),'cell')
        val    = char(val{:});
      end
      if size(val,1)>1
        error(['  Size(value,1) (or number of cell elements) must ', ...
          '= 1 for field ''title''.'])
      end
      
      nsets = size(x.title,3);
      if nsets<vset;
        [x.title{:,:,nsets+1:vset}] = deal('');   %expand to new set(s) with empty STRINGS
      end
      x.title{vdim,1,vset}= val;
      
      [x.history,ts] = sethistory(x.history,'','title',['''' val ''''],vdim,vset);
      
    case 'titlename'
      if ~isempty(index);
        error(['  Field ''' feyld ''' does not allow assignment subscripting']);
      end;
      if isempty(x.data)
        error(['  ''' feyld ''' can not be set when ''data'' is empty.'])
      elseif isempty(val)
        val = char(val);
      elseif ~(strcmp(class(val),'char')|strcmp(class(val),'cell'))
        error('  Value must be class char or cell for field ''titlename''.')
      elseif strcmp(class(val),'cell')
        val    = char(val{:});
      end
      if size(val,1)>1
        error(['  Size(value,1) (or number of cell elements) must ', ...
          '= 1 for field ''titlename''.'])
      end
      
      nsets = size(x.title,3);
      if nsets<vset;
        [x.title{:,:,nsets+1:vset}] = deal('');   %expand to new set(s) with empty STRINGS
      end
      x.title{vdim,2,vset}= val;
      
      [x.history,ts] = sethistory(x.history,'','titlename',['''' val ''''],vdim,vset);
      
    case 'class'
      %Intercept assignment of strings.
      % E.G., mydso.class{1,2} = {'sdfsd' 'sdfds' 'sdfds'}
      if ischar(val)
        if size(val,1)==1;
          %May be assigning single value.
          % E.G., mydso.class{1,2}(6) = 'ssdf';
          val = {val};
        else
          %multi-line string, sort into cells
          temp = cell(1,size(val,1));
          for j=1:size(val,1);
            temp{j} = val(j,:);
          end
          val = deblank(temp);
        end
      end
      if iscellstr(val)
        if size(val,1)>size(val,2)
          %Transpose so direction is correct for later assignments.
          val = val';
        end
        %Get existing lookup table if there is one.
        try
          mytbl = x.classlookup{vdim,vset};
        catch
          mytbl = [];
        end
        %Convert string values to numeric classes and configure lookup
        %table.
        [mytbl val] = stringtable(val,mytbl);
        x.classlookup{vdim,vset} = mytbl;
        
        %If indexing outside of array Matlab will default infill with []
        %(empty matrix) so change any empty matrices to empty cells.
        for ci = 1:numel(x.classlookup)
          if isempty(x.classlookup{ci})
            x.classlookup{ci} = {};
          end
        end
        
      end
      
      if isnumeric(val)
        % Convert any NaN class values to zero values
        i = find(isnan(val));
        if any(i)
          val(i) = 0;
          warning('EVRI:DSOClassNaN','DataSet class cannot be NaN. Using "zero" to indicate unknown class.');
        end
      end
      
      if ~isempty(index);
        if vset>size(x.class,3);
          error(['  ''class'' set ' num2str(vset) ' does not exist.'])
        end
        if strcmpi(index.type,'{}');
          %if they used cell indexing with a string, we've alrady converted
          %the string to a number by this point, so just use () indexing
          index.type = '()';
        end
        try
          val = subsasgn(x.class{vdim,1,vset},index,val);
          %If val is anything other than set length then output from
          %subsasgn will return a vector that's too short.
          %Pad output with zero vector.
          if length(val)<size(x,vdim)
            val(size(x,vdim)) = 0;
          end
        catch; error(lasterr); end;
      end;
      if isempty(x.data)
        error(['  ''' feyld ''' can not be set when ''data'' is empty.'])
      end
      if (size(val,1)>1&size(val,2)>1)|(ndims(val)>2)|(~strcmp(class(val),'double'))
        error('  Value must be vector of class double.')
      end
      if ~isa(x.data,'cell') | vdim == 1;
        if size(x.data,vdim)~=length(val) & ~isempty(val);
          error(['  Length of input ''class'' must =', ...
            'size(x.data,vdim) = ',int2str(size(x.data,vdim)),'.'])
        end
      else
        if size(x.data{1},vdim)~=length(val) & ~isempty(val);
          error(['  Length of input ''class'' must =', ...
            'size(x.data{1},vdim) = ',int2str(size(x.data{1},vdim)),' (for vdim > 1).'])
        end
      end
      if size(val,1)>size(val,2)
        val  = val';
      end
      
      nsets = size(x.class,3);
      if nsets<vset;
        [x.class{:,2,nsets+1:vset}] = deal('');   %expand to new set(s) with empty STRINGS
      end
      x.class{vdim,1,vset} = val;
      
      %TODO: Add an update to lookup table if a numeric only class
      %asignment occures. mydso.class{1,2} = [2 2 0 5 1 1 2];
      
      
      %Create lookup table if doesn't exist.
      if size(x.classlookup,1)<vdim || size(x.classlookup,2)<vset || isempty(x.classlookup{vdim,vset})
        x.classlookup{vdim,vset} = classtable(val);
        %If indexing outside of array Matlab will default infill with []
        %(empty matrix) so change any empty matrices to empty cells.
        for ci = 1:numel(x.classlookup)
          if isempty(x.classlookup{ci})
            x.classlookup{ci} = {};
          end
        end
      else
        %There is an existing classlookup table so update it with new
        %numeric class is necessary.
        x.classlookup{vdim,vset} = classtable(val,x.classlookup{vdim,vset});
      end
      
      notes = ['class double ', int2str(size(val,1))];
      for ii=2:ndims(val)
        notes   = [notes,'x',int2str(size(val,ii))];
      end
      if ~isempty(index);
        notes = [notes ' (subscript replacement)'];
      end;
      [x.history,ts] = sethistory(x.history,'','class','',vdim,vset,notes);
      
    case 'classid'
      idx(1).type = '.';
      idx(1).subs = 'class';
      idx(2).type = '{}';
      idx(2).subs = { vdim vset };
      if ~isempty(index)
        idx = [idx index];
      end
      x=subsasgn(x,idx,val);
    case 'classname'
      if ~isempty(index);
        error(['  Field ''' feyld ''' does not allow assignment subscripting']);
      end;
      if isempty(x.data)
        error(['  ''' feyld ''' can not be set when ''data'' is empty.'])
      elseif isempty(val)
        val = char(val);
      elseif ~(strcmp(class(val),'char')|strcmp(class(val),'cell'))
        error('  Value must be class char or cell for field ''classname''.')
      elseif strcmp(class(val),'cell')
        val    = char(val{:});
      end
      if size(val,1)>1
        error(['  Size(value,1) (or number of cell elements) must ', ...
          '= 1 for field ''classname''.'])
      end
      
      %Create lookup table if doesn't exist.
      if size(x.classlookup,1)<vdim || size(x.classlookup,2)<vset || isempty(x.classlookup{vdim,vset})
        x.classlookup{vdim,vset} = {};
        %If indexing outside of array Matlab will default infill with []
        %(empty matrix) so change any empty matrices to empty cells.
        for ci = 1:numel(x.classlookup)
          if isempty(x.classlookup{ci})
            x.classlookup{ci} = {};
          end
        end
      end
      nsets = size(x.class,3);
      if nsets<vset;
        [x.class{:,2,nsets+1:vset}] = deal('');   %expand to new set(s) with empty STRINGS
      end
      x.class{vdim,2,vset}= val;
      
      [x.history,ts] = sethistory(x.history,'','classname',['''' val ''''],vdim,vset);
      
    case 'classlookup'
      %Mode 1 should contain elements coresponding to dimensionality of data.
      %Mode 2 should contain elements corresponding to lookup "sets" for that dimension.
      %Each element should contain a single lookup table (n x 2 cell array).
      %First column should numeric values.
      %Second column should contain stings (names for each class).
      %These columns should be equal in length.
      %There may be dynamic list changes going on so:
      %  Can have partial lookup tables.
      %  No checking for existence of numeric class, may not exist yet.
      
      %Can either index into table or assign entire table.
      
      %Allow name/value changes for given values.
      %TODO: Assess if we want to allow normal indexing changes.
      %E.G., arch.classlookup{1}(2,:) = {7 'Class 7'}
      %E.G., arch.classlookup{1}(2,1) = {7}
      if ~isempty(index);
        nval = x.classlookup{vdim,vset};
        %Put inputs into correct order with numeric class first so user can
        %input either way .assignstr = {1 'blah'} or .assignstr = {'blah' 1}.
        if ~isnumeric(val{1})
          val = fliplr(val);
        end
        if strcmp(index.subs,'assignstr')
          %Using special 'assign' to change string for given numeric class.
          %E.G., arch.classlookup{2,2}.assignstr = {1 'blah'}
          if size(val,1)>1;
            error('Can only assign one string at a time (expected single row cell array {value ''string''} )')
          elseif size(val,2)~=2
            error('Valid input is a class number and a single string to assign to the given class');
          end
          row = find(ismember([nval{:,1}],val{1}));
          if isempty(row)
            error(['Can''t find class: ' num2str(val{1}) ' in classlookup table.']);
          end
          nval{row,2} = val{2};
        elseif strcmp(index.subs,'assignval')
          %Using special 'assign' to change numeric class for given string.
          %E.G., arch.classlookup{2,2}.assignval = {4 'blah'}
          if size(val,1)>1;
            error('Can only assign one class value at a time (expected single row cell array {value ''string''} )')
          elseif size(val,2)~=2
            error('Valid input is a class value and a single string to which the given class value should be assigned');
          end
          row = find(ismember(nval(:,2),val{2}));
          if isempty(row)
            error(['Can''t find classname: ' val{2} ' in classlookup table.']);
          end
          oldrow = nval(row,:);
          nval{row,1} = val{1};
          %Change old numeric class to new class.
          cls = x.class{vdim,1,vset};
          cls(ismember(cls,oldrow{1})) = val{1};  %re-assign as class 0
          x.class{vdim,1,vset} = cls;
        else
          error('valid classlookup methods are "assignval" and "assignstr"');
        end
        val = nval;
        index = [];
      end
      
      if ~isempty(index);
        if vset>size(x.classlookup,2);
          %Index into table, check set dimension (mode 2).
          error(['  ''class'' set ' num2str(vset) ' does not exist.'])
        end
        try
          %Call to directly assign value.
          val = subsasgn(x.classlookup{vdim,vset},index,val);
          %TODO: History more explicit.
        catch; error(lasterr); end;
      end;
      if isempty(x.data)
        error(['  ''' feyld ''' can not be set when ''data'' is empty.'])
      end
      
      %No mode 1 size checks, user may be creating and or deleting classes
      %dynamically.
      
      if ~isempty(val) && size(val,2)~=2
        %Check for nx2.
        error(['Class lookup table must have 2 columns.'])
      end
      
      if ~isempty(val) & (any(cellfun('isempty',val(:,1))) | any(cellfun(@(i) ~isnumeric(i),val(:,1),'uniformoutput',true)))
        %Check for numeric first column
        error('First column in class lookup table must contain non-empty numeric class identifiers.')
      end      
            
      if ~isempty(val) & any(cellfun(@(i) ~ischar(i),val(:,2),'uniformoutput',true))
        %Check for char second column.
        error(['Second class lookup table column must be character type data.'])
      end
      
      if ~isempty(val) && length(unique(cat(2,val{:,1})))~=length(cat(2,val{:,1}))
        %Check for duplicate numerical entries
        error(['Class lookup table classes must be unique.'])
      end
      
      %Expand sets.
      nsets = size(x.classlookup,2);
      if nsets<vset;
        [x.classlookup{:,nsets+1:vset}] = deal('');   %expand to new set(s) with empty STRINGS
      end
      
      nsets_class = size(x.class,3);
      if (nsets_class>= vset && ~isempty(x.class{vdim,1,vset})) && (isempty(val) || ~ismember(0,cat(2,val{:,1})))
        %no class zero? ALWAYS add one (as long as we have classes)
        val = [{0 'Class 0'};val];
      end
      
      if vset>nsets_class
        %Add empty class sets if they don't exist.
        idx(1).type = '.';
        idx(1).subs = 'class';
        idx(2).type = '{}';
        idx(2).subs = { vdim vset };
        x=subsasgn(x,idx,[]);
      end
      
      %get old lookup and check for missing classes
      oldlookup = x.classlookup{vdim,vset};
      if isempty(val)
        %nothing in new set, everything in old set is "missing"
        if ~isempty(oldlookup)
          missing = cat(2,oldlookup{:,1});
        else
          missing = []; %nothing there, nothing new, nothing missing
        end
      elseif isempty(oldlookup)
        %nothing in oldlookup but newlookup has something, nothing missing
        missing = [];
      else
        %something in both, do setdiff
        missing = setdiff(cat(2,oldlookup{:,1}),cat(2,val{:,1}));
      end
      if ~isempty(missing)
        cls = x.class{vdim,1,vset};
        cls(ismember(cls,missing)) = 0;  %re-assign as class 0
        x.class{vdim,1,vset} = cls;
      end
      
      %sort (if not empty)
      if ~isempty(val);
        [junk,order] = sort(cat(2,val{:,1}));
        val = val(order,:);
      end
      
      %Add new lookup
      x.classlookup{vdim,vset} = val;
      
      %If indexing outside of array Matlab will default infill with []
      %(empty matrix) so change any empty matrices to empty cells.
      for ci = 1:numel(x.classlookup)
        if isempty(x.classlookup{ci})
          x.classlookup{ci} = {};
        end
      end
      
      notes = ['cell array ', int2str(size(val,1))];
      for ii=2:ndims(val)
        notes   = [notes,'x',int2str(size(val,ii))];
      end
      if ~isempty(index);
        notes = [notes ' (subscript replacement)'];
      end;
      [x.history,ts] = sethistory(x.history,'','classlookup','',vdim,vset,notes);
      
    case 'description'
      error('  Indicies not allowed for field ''description''.')
      
    case {'includ','include'}
      if vset>1;
        error(['  Field ''' feyld ''' does not allow multiple sets']);
      end;
      if ~isempty(index);
        try
          val = subsasgn(x.include{vdim},index,val);
        catch; error(lasterr); end;
      end;
      if ~isa(x.data,'cell')  | vdim == 1
        nmodes  = ndims(x);
        nsize   = size(x);
      else
        nmodes  = ndims(x.data{1});
        nsize   = size(x.data{1});
        nsize(1)= length(x.data);
      end
      if any(val<1)|(max(val)>nsize(vdim))
        error(['  Value must be row vector of integers and a', ...
          ' subset of [1:',int2str(nsize(vdim)),'].'])
      elseif size(val,1)>1&size(val,2)>1
        error('  Value must be row vector.')
      else
        if size(val,1)>size(val,2)
          val   = val';
        end
        
        if length(val)==length(x.include{vdim}) & (isempty(val) | all(val==x.include{vdim}))
          %if they aren't actually changing the include field... just
          %return it now without making any changes to history or date
          return;
        end
        x.include{vdim} = val;
        
        notes = ['class double ', int2str(size(val,1))];
        for ii=2:ndims(val)
          notes   = [notes,'x',int2str(size(val,ii))];
        end
        if ~isempty(index);
          notes = [notes ' (subscript replacement)'];
        end;
        [x.history,ts] = sethistory(x.history,'','include','',vdim,vset,notes);
      end
    case 'userdata'
      error('  Indicies not allowed for field ''userdata''.')
    otherwise
      error('  Not valid dataset object field.')
  end
  if isempty(ts)
    ts = clock;
  end
  if isempty(x.date)
    x.date = ts;
  end
  x.moddate = ts;
  
elseif nnargin>5
  error('Too many inputs.')
end

%--------------------------------------------------------------------------
function [myhistory,ts] = sethistory(myhistory,objname,feyld,val,vdim,vset,notes)
%Add history log entry.

%NOTE: As of 2015b 'inputname' is not allowed in subsasgn (and will cause
%error). Removed all usage of it prior to 15b release so 'objname' is
%always empty. Keeping 'objname' input for now in case we want to re-enable
%the functionality (somehow) in future but may choose to remove it as well.

if isempty(myhistory{1})
  ihis   = 1;
else
  ihis   = length(myhistory)+1;
end

caller = '';
try
  [ST,I] = dbstack;
  if length(ST)>2;
    [a,b,c]=fileparts(ST(3).name);
    caller = [' [' b c ']'];
  end
catch
end

if isempty(val);
  val = '?FunctionCall';
end

if nargin>6;
  notes  = [' - ' notes caller];
else
  notes  = [' - ' caller ];
end;

[tstamp,ts] = timestamp;
switch feyld
  case {'name', 'author', 'type', 'description', 'data', 'userdata','imagesize','imagemode','comment'}
    myhistory{ihis} = [objname '.' feyld '=' val '; % ' tstamp notes];
  otherwise
    if nargin>4 & (~isempty(vdim) | ~isempty(vset))
      myhistory{ihis} = [objname '.' feyld '{' int2str(vdim) ',' int2str(vset) '}=' val '; % ' tstamp notes];
    else
      myhistory{ihis} = [objname '.' feyld '=' val '; % ' tstamp notes];
    end
end

