function value=subsref(x,index)
%DATASET/SUBSREF Subindex for DataSet objects using structure and index notation.
%I/O: x = x(rows,cols);
%  reduces the DataSet object to the specified rows and columns (all fields
%  are indexed)
%I/O: x = x{batch};
%  when (x) is a 'batch' type DSO, this extracts the specified batch
%  ((batch) must be a scalar value) and converts the batch to a standard
%  'data' type DataSet object.
%I/O: value = x.field;
%  when (x) is a dataset object (See DATASET) this returns
%  the value (value) of the dataset object field ('field').
%  This syntax is used for the following fields:
%    name           : char vector.
%    author         : char vector.
%    date           : 6 element vector (see CLOCK).
%    moddate        : 6 element vector (last modified date).
%    type           : either 'data', 'batch' or 'image'.
%    data           : double,single, logical, or [u]int8/16/32 array.
%    data.include   : included data, can be indexed with include modes.
%    table          : return data with row and column labels.
%    size           : numeric vector.
%    sizestr        : char vector.
%    imagedata      : multi-way array.
%    imagedataf     : multi-way array with spacial data starting at imagemode.
%    imagesize      : vector no longer than ndims of data.
%    imagesizestr   : char vector.
%    imagemode      : scalar.
%    foldedsize     : numeric vector.
%    foldedsizestr  : char vector.
%    label          : cell of char arrays.
%    labelname      : cell of char row vectors.
%    axisscale      : cell of row vectors.
%    imageaxisscale : cell of row vectors.
%    axisscalename  : cell of char row vectors.
%    imageaxisscalename : cell of char row vectors.
%    axistype       : cell of char row vectors, must be ['discrete'|'stick'|'continuous'|'none'].
%    imageaxistype  : cell of char row vectors, must be ['discrete'|'stick'|'continuous'|'none'].
%    title          : cell of char row vectors.
%    titlename      : cell of char row vectors.
%    class          : cell of row vectors.
%    classname      : cell of char row vectors.
%    classlookup    : cell of nx2 where col 1 is double and col2 is string.
%    classid        : a reference that assigns/returns a cell array of stings
%                     based on the .classlookup table.
%    userdata       : user defined.
%    uniqueid       : char vector.
%    description    : char or cell array.
%    history        : cell array of char (e.g. char(x.history))
%    datasetversion: dataset object version.
%
%I/O: value = x.field{vdim};
%    include      : value is row vector of indices for mode vdim
%
%I/O: value = x.field{vdim,vset};
%  when (x) is a dataset object this returns the value (value)
%  for the field ('field') for the specified dimension/mode
%  (vdim) and optional set (vset) {default: vset=1}. E.g. (vset) is
%  used when multiple sets of labels are present in (x).
%  This syntax is used for the following fields:
%    label          : value is a char array with size(x.data,vdim) rows.
%    labelname      : value is a char row vector.
%    axisscale      : value is a row vector with size(x.data,vdim) real elements.
%    imageaxisscale : value is a row vector with x.imagemode(vdim) real elements.
%    axisscalename  : value is a char row vector.
%    imageaxisscalename: value is a char row vector.
%    title          : value is a char row vector.
%    titlename      : value is a char row vector.
%    class          : value is a row vector with size(x.data,vdim) integer elements.
%    classname      : value is a char row vector.
%
%Examples:
%    plot(mydataset.axisscale{2})            %second dim axis scale
%    xlabel(mydataset.title{1,2})            %Second title for first dim
%    disp(['Made by: ' mydataset.author])
%
%See also: DATASET, DATASETDEMO, DATASET/SUBSASGN, DATASET/GET, ISA, DATASET/EXPLODE

%Copyright Eigenvector Research, Inc. 2000-2019
%nbg 8/3/00, 8/16/00, 8/17/00, 8/30/00, 9/1/00, 10/10/00
%jms 4/11/01  -converted from dataset/get
%   -added indexing into history (vs. get which doesn't allow that)
%jms 5/3/01 -added indexing into data using ()s (e.g. x.data(1:5,:) )
%jms 5/9/01 -adapted for use with data being other classes (single, uint8, etc).
%jms 5/31/01 -added subscripting into ALL fields
%jms 8/02 -allow extract of all labels, axisscales, etc at once by calling
%           without vdim
%  -revised error messages
%jms 9/02 -added object-level subscripting
%jms 4/24/03 -renamed "includ" to "include"
%jms 11/7/03 -base vdim tests on ndims(x) code instead of ndims(x.data)
%rsk 2/05 -added image fields
%jms 10/31/05 -allow cell indexing on userdata

%Fields 'label', 'axisscale', 'title', and 'class' can not be set if 'data'
%is empty. Once field 'data' has been filled it can not be "set" to a 
%different size.

nnargin = nargin;
error(nargchk(2,2,nnargin));   %give error if nargin is not appropriate

if isempty(index); error('  Invalid subscripting'); end;

if ~strcmp(index(1).type,'.')
  %No field reference, return DSO of indexed data.
  switch index(1).type
    case '()'
      for dim = 1:length(index(1).subs)
        %convert logical to double so we can test for out of range
        if isa(index(1).subs{dim},'logical')  
          index(1).subs{dim} = find(index(1).subs{dim});
        end
      end
      if length(index(1).subs)==1 & size(x.data,1)==1 & size(x.data,2)>1
        %when using single index on ROW vector, change index to second
        %mode
        index(1).subs = {1 index(1).subs{1}};
      end
      for dim = 1:length(index(1).subs)
        %interpret doubles as indexing into dataset object (do delsamps on UNspecified indices)
        if isa(index(1).subs{dim},'double')
          if max(index(1).subs{dim}) > size(x.data,dim);
            error('  Invalid subscript')
          end
          if dim <= ndims(x)   %don't try this if we don't have the dim
            inds = index(1).subs{dim};
            x = delsamps(x, inds(:), dim, 3);    %hard keep/reorder delete
          end
        end
      end
      
      %********************************************************************
      % Added section for calls like: dso('classname').include; & dso('classname').exclude;
      % Works for n-way data, on any mode.
      if (isa(index(1).subs(:),'cell') & (length(index(1).subs(:))<2)) % This check needs to be here or CV won't work!
        if (isa(index(1).subs{:},'char')) % Check if index(1).subs is a string (for reasons of compatibility).
          % Check if the input string matches a classname
          classLookup = x.classlookup;
          for i = 1:size(classLookup,2)
            for j = 1:size(classLookup,1)
              if ~(isempty(classLookup{j,i}))
                classVals = classLookup{j,i}(:,1);
                classNames = classLookup{j,i}(:,2);
                for k = 1:size(classNames,1) % Check the class names in the sets
                  if strcmpi(index(1).subs{:},classNames{k,1})
                    % Input matches a classname, retrieve all indexes of that class.
                    if (ndims(x.class)==3) % classes have 3 modes
                      classIdxs = find(x.class{j,1,i}==classVals{k});
                    else % Classes have 2 modes
                      classIdxs = find(x.class{j,i}==classVals{k});
                    end
                    useIdx = classIdxs;
                    if (length(index)>1) % Did the user use '.include'/'.exclude'? Get the corresponding indexes.
                      if strcmpi(index(2).subs,'include') | strcmpi(index(2).subs,'exclude')
                        if strcmpi(index(2).subs,'include')
                          incIdx = x.include{j};
                        elseif strcmpi(index(2).subs,'exclude')
                          incIdx = setxor(1:size(x,j),x.include{j});
                        end
                        useIdx = intersect(useIdx,incIdx); % Get all the indexes to remove
                      else
                          error(sprintf(['.' index(2).subs ' is not a valid operator in the context of indexing a class by name']));
                      end
                    end
                    value = delsamps(x,setxor(1:size(x,j),useIdx),j,2);
                    return;
                  end
                end
              end
            end
          end
          % If the code reaches here, the entered class name could not be found.
          %error(sprintf(['Could not find the ' index(1).subs{:} ' class in dataset.']));
        end
      end
      %********************************************************************
      
      index(1) = [];
      if isempty(index);    %nothing other than indexing into main object? just return reduced object
        value = x;
        return
      end
      
    case '{}'
      %handle cell-array indexing on top-level object (converts batch to
      %data type)
      if ~strcmp(x.type,'batch')
        error('  Cell indexing only valid with ''batch'' type DataSet obejcts');
      end
      index(1).type = '()';
      x = subsref(x,index(1));  %do indexing as () first (uses code above)
      if length(x.data)==1
        %only one cell left? convert to data
        x.type = 'data';
        x.axisscale{1} = x.axisscale{1}{1};
        x.data = x.data{1};
        x.include{1} = 1:size(x,1);
        x.history{length(x.history)} = ['cell {',num2str(index(1).subs{1}),'} %extract from type batch'];

      else
        error('   Cell indexing into ''batch'' type DataSet must extract only one batch');
      end        
      
      index = index(2:end);
      if isempty(index)
        value = x;
        return;
      end
      
    otherwise
      error('  Invalid subscripting for DataSet object');
  end
end
feyld = index(1).subs; %Field name.

%NOTE: for copatibility with The Mathworks DSO, the following subsitutions
%would need to be done:
%     Description: ''                             %x.description
%        UserData: []                             %userdata
%           Units: {'a1'  'a2'  'a3'}             %repmat(x.axisscalename(2),1,n)
%        DimNames: {'Observations'  'Variables'}  %x.title'
%        ObsNames: {2x1 cell}                     %str2cell(x.label{1})
%        VarNames: {'One'  'TwoAnd'  'Three'}     %str2cell(x.label{2})'
% Also note that indexing into TMWDSO with a variable name:
%    x.Three
% would be the same as an EVRI DSO operation:
%    x.Three.data

if size(feyld,1)>1
  error('  Input ''field'' must be char row vector or 1 element cell.')
elseif isa(feyld,'cell')&size(feyld,2)>1
  error('  Input ''field'' must be char row vector or 1 element cell.')
end
if isa(feyld,'cell')&~isempty(feyld)
  feyld  = char(feyld{:});
end
indx     = '';
if strncmpi(feyld,'data',4) & length(feyld)>4
  %Index into data field. Parse remaining characters into index becuase
  %remaining characters make up index information.
  indx   = feyld(1,5:end);
  feyld  = 'data';
end

if length(index)<2 | ~strcmp(index(2).type,'{}') | ...
    (strcmp(x.type,'batch') & strcmpi(feyld,'data')) ...
    | strcmpi(feyld,'userdata');
  %Make set and dim to one so retrieve entire field in nnargin=2 case.
  %Calls without vdim or vset (e.g. on data, date, author, etc)
  vdim  = 1;
  vset  = 1;
  nnargin = 2;     %fake out logic later on
  index(1) = [];  %clear field name from index structure
else
  %Indexing into specific location.
  vdim=index(2).subs{1};
  if length(index(2).subs)>1;
    vset=index(2).subs{2};
    nnargin = 4;     %fake out logic later on
  else
    vset=1;
    nnargin = 3;     %fake out logic later on
  end;
  index(1:2) = [];  %clear field name & vset/vdim from index structure
  %Remaining parts of index will be used to subscript into results after
  %getting results.
end;

if nnargin==2
  switch lower(feyld)
  case 'name'
    value  = x.name;
  case 'author'
    value  = x.author;
  case 'date'
    value  = x.date;
  case 'moddate'
    value  = x.moddate;
  case 'type'
    value  = x.type;
  case 'size'
    value = size(x);
  case 'sizestr'
    sz = size(x);
    sizestr = sprintf('%ix',sz);
    value = [sizestr(1:end-1)];
  case 'imagesize'
    if strcmp(x.type,'image')
      value  = x.imagesize;
    else
      value  = [];
    end
  case 'imagesizestr'
    if strcmp(x.type,'image')
      sz  = x.imagesize;
      sizestr = sprintf('%ix',sz);
      value = [sizestr(1:end-1)];
    else
      value  = [];
    end
  case 'foldedsize'
    if strcmp(x.type,'image')
      insize = size(x);
      indims = 1:ndims(x);
      modloc = find(ismember(indims,x.imagemode));
      value = [insize(1:modloc-1) x.imagesize insize(modloc+1:end)];
    else
      value  = [];
    end
  case 'foldedsizestr'
    if strcmp(x.type,'image')
      idx(1).type = '.';
      idx(1).subs = 'foldedsize';
      sz = subsref(x,idx); 
      sizestr = sprintf('%ix',sz);
      value = [sizestr(1:end-1)];
    else
      value  = [];
    end
  case 'imagemode'
    if strcmp(x.type,'image')
      value  = x.imagemode;    
    else
      value  = [];
    end
  case 'imagemap'
    if strcmp(x.type,'image')
      value  = displayincludemap(x);    
    else
      value  = [];
    end
  case 'imagedata'
    %Image data with spatial first.
    if strcmp(x.type,'image')
      value  = displayimage(x);    
    else
      value  = [];
    end
  case 'imagedataf'
    %Image data with spatial at .imagemode.
    if strcmp(x.type,'image')
      value  = displayimage(x,'on');    
    else
      value  = [];
    end
  case 'data'
    %Test for include indexing.
    if ~isempty(index) & strcmp(index(1).type,'.') & ismember(lower(index(1).subs),{'include' 'includ'})
      S      = [];
      S.type = '()';
      S.subs = cell(1,ndims(x));
      [S.subs{1:end}] = deal(':');   % gives  {':' ':' ':' ...}
      if length(index)>1
        mydims = index(2).subs{:};
      else
        mydims = 1:ndims(x);
      end
      
      for i = mydims
        index(1).subs = 'include';
        index(2).type='{}';
        index(2).subs={i};
        include_idx = subsref(x,index);
        S.subs{i} = include_idx;
      end
      
      if anyexcluded(x)
        value = subsref(x.data,S);
      else
        %nothing excluded - just grab data (faster, more memory friendly)
        %NOTE: we go ahead and do all the logic above (even if we've got
        %nothing excluded) so that we can throw an error if the user
        %screwed up the call. That way, testing with a non-excluded data
        %will show you something is wrong even if there is nothing to
        %exclude.
        value = x.data;
      end
      
      %Clear index so won't run through again.
      index(1:2) = [];
    else
      if isempty(indx)
        value  = x.data;
      else
        %special case where additional index info is passed in the field
        %name (instead of in the index vector as usual)
        eval(['value = x.data',indx,';'])
      end
    end
  case 'table'
    value  = copy_clipboard(x);
  case 'description'
    value  = x.description;
  case 'datasetversion'
    value  = x.datasetversion;
  case 'userdata'
    value  = x.userdata;
  case 'history'
    value  = x.history;
  case 'label'
    value  = squeeze(x.label(:,1,:));
  case 'labelname'
    value  = squeeze(x.label(:,2,:));
  case 'axisscale'
    value  = squeeze(x.axisscale(:,1,:));
  case 'imageaxisscale'
    value  = squeeze(x.imageaxisscale(:,1,:));
  case 'axisscalename'
    value  = squeeze(x.axisscale(:,2,:));
  case 'imageaxisscalename'
    value  = squeeze(x.imageaxisscale(:,2,:));
  case 'axistype'
    value  = squeeze(x.axistype(:,:));
  case 'imageaxistype'
    value  = squeeze(x.imageaxistype(:,:));
  case 'title'
    value  = squeeze(x.title(:,1,:));
  case 'titlename'
    value  = squeeze(x.title(:,2,:));
  case 'class'
    value  = squeeze(x.class(:,1,:));
  case 'classid'
    value = '';
    %Loop through dims and sets to contruct classid.
    for i = 1:size(x.class,1)
      for j = 1:size(x.class,3)
        idx(1).type = '.';
        idx(1).subs = 'classid';
        idx(2).type = '{}';
        idx(2).subs = { i j };
        value{i,j} = subsref(x,idx); 
      end
    end
  case 'classlookup'
    value  = squeeze(x.classlookup(:,:));
  case 'classname'
    value  = squeeze(x.class(:,2,:));
  case {'includ','include'}
    value  = x.include(:,1)';
  case 'uniqueid'
    value = x.uniqueid;
  otherwise
    %check if this is a valid label (start with columns, then check rows)
    [value,match] = findlabelmatch(x,feyld);
    if isempty(match);
      error('  Not valid dataset object field.')
    end
  end
elseif nnargin==3|nnargin==4
  if vdim(1)<1
    error(['vdim(1) must be integer >0.'])
  elseif any(strcmpi(feyld,{'history'}));    %allow : expansion on these fields
    if isa(vdim,'char');
      if ~strcmp(vdim,':');
        error(['expecting integer index or '':'' operator']);
      else
        vdim=1:length(getfield(x,feyld));
      end;
    elseif((max(vdim)>length(x.history))|min(vdim)<1)
      error(['vdim must be integer in [1:',int2str(length(getfield(x,feyld))),'].'])
    end;
  else
    switch x.type
      case 'batch'
        if (vdim(1)>ndims(x.data{1}))
          error(['vdim(1) must be integer in [1:',int2str(ndims(x.data{1})),'].'])
        elseif (length(vdim)>2)
          error(['vdim must be 2 element vector.'])
        elseif (length(vdim)==2)&((vdim(2)>length(x.data))|(vdim(2)<1))
          error(['vdim(2) must be integer in [1:',int2str(length(x.data)),'].'])
        end
        
      case 'image'
        if strcmpi(feyld,'imageaxisscale') 
          if ((vdim>length(x.imagesize))|length(vdim)>1)
            error(['vdim must be integer in [1:',int2str(length(x.imagesize)),'].'])
          end
        else
          %other fields
          if ((vdim>ndims(x))|length(vdim)>1) 
            error(['vdim must be integer in [1:',int2str(ndims(x)),'].'])
          end
        end
      otherwise
        if ((vdim>ndims(x))|length(vdim)>1)
          error(['vdim must be integer in [1:',int2str(ndims(x)),'].'])
        end
      
    end
    if vset<1
      error('Input vset can not be less than zero.')
    end
  end;
  
  nlabel     = size(x.label,3);     %get number of sets for each
  naxisscale = size(x.axisscale,3); %field
  nimageaxisscale = size(x.imageaxisscale,3); %field
  ntitle     = size(x.title,3);
  nclass     = size(x.class,3);
  nclass_lu  = size(x.classlookup,2);
  switch lower(feyld)
  case {'name' 'author' 'date' 'moddate' 'type' 'imagesize' 'imagemode' 'data' 'description' 'datasetversion' 'userdata' 'uniqueid'}
    error('  Indexing not allowed for field ''%s''.',lower(feyld))
  case 'history'
    if length(vdim)==1;
      value  = x.history{vdim};
    else
      value  = char(x.history(vdim));
    end;
  case 'label'
    if vset>nlabel
      value = '';
    else
      value  = x.label{vdim,1,vset};
    end
  case 'labelname'
    if vset>nlabel
      value = '';
    else
      value  = x.label{vdim,2,vset};
    end
  case 'axisscale'
    if vset>naxisscale
      value = [];
    else
      if isa(x.data,'cell')
        if (length(vdim)==1)
          value  = x.axisscale{vdim,1,vset};
        elseif (length(vdim)==2)&(vdim(1)==1)
          value  = x.axisscale{1,1,vset}{vdim(2)};
        else
          error('vdim must be a 2 element vector with vdim(1)== 1');
        end;
      else
        value    = x.axisscale{vdim,1,vset};
      end
    end
  case 'imageaxisscale'
    if vset>nimageaxisscale
      value = [];
    else
      if isa(x.data,'cell')
        if (length(vdim)==1)
          value  = x.imageaxisscale{vdim,1,vset};
        elseif (length(vdim)==2)&(vdim(1)==1)
          value  = x.imageaxisscale{1,1,vset}{vdim(2)};
        else
          error('vdim must be a 2 element vector with vdim(1)== 1');
        end;
      else
        value    = x.imageaxisscale{vdim,1,vset};
      end
    end
  case 'axisscalename'
    if vset>naxisscale
      value = '';
    else
      value  = x.axisscale{vdim,2,vset};
    end
  case 'imageaxisscalename'
    if vset>nimageaxisscale
      value = '';
    else
      value  = x.imageaxisscale{vdim,2,vset};
    end
  case 'axistype'
    if vset>naxisscale
      value = 'none';
    else
      value  = x.axistype{vdim,vset};
    end
  case 'imageaxistype'
    if vset>nimageaxisscale
      value = 'none';
    else
      value  = x.imageaxistype{vdim,vset};
    end
  case 'title'
    if vset>ntitle
      value = '';
    else
      value  = x.title{vdim,1,vset};
    end
  case 'titlename'
    if vset>ntitle
      value = '';
    else
      value  = x.title{vdim,2,vset};
    end
  case 'class'
    if vset>nclass
      value = [];
    else
      value  = x.class{vdim,1,vset};
    end
  case 'classlookup'
    if vset>nclass_lu
      value = {};
    elseif ~isempty(index) & strcmp(index(1).subs,'find')
      %Find string for particular value.
      %arch.classlookup{1,1}.findstr(4)
      if (~isnumeric(index(2).subs{:}) || max(size(index(2).subs{:}))~=1) &&...
          ~ischar(index(2).subs{:})
        error('Index to classlookup.find must be a single numeric or string value.')
      end
      mytable = x.classlookup{vdim,vset};
      %Find row/value where number is located.
      if ischar(index(2).subs{:})
        row = find(ismember(mytable(:,2),index(2).subs{:}));
        value = mytable(row,1); %Extract class value for row.
      else
        row = find(ismember([mytable{:,1}],index(2).subs{:}));
        value = mytable(row,2); %Extract class name for row.
      end
      %Clear index so won't run through again.
      index(1:2) = [];
    else
      %Extract the entire table.
      value  = x.classlookup{vdim,vset};
    end
  case 'classid'
    %Returns cell array of strings for class.
    if vset>nclass
      value = {};
    else
      clsmap  = x.classlookup{vdim,vset};
    end
    
    cls = x.class{vdim,1,vset};
    if isempty(clsmap)
      value = {};
    else
      %Create a cell array of empties so that if the lookup table doesn't
      %have an entry for a particular class it will show up as an empty.
      value = repmat({''},1,length(cls));
      for ic = 1:size(clsmap,1)
        cinds = ismember(cls,clsmap{ic,1});
        value(cinds) = clsmap(ic,2);
      end
    end
    
  case 'classname'
    if vset>nclass
      value = '';
    else
      value  = x.class{vdim,2,vset};
    end
  case {'includ','include'}
    value    = x.include{vdim};
  otherwise
    [value,match] = findlabelmatch(x,feyld);
    if isempty(match);
      error('  Not valid dataset object field.')
    end
  end
end

if ~isempty(index);
  %some indexing instructions left in index
  try;
    value = subsref(value,index);
  catch
    error(lasterr)
  end;
end;
  

