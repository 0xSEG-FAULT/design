function y = cat(dim,varargin);
%DATASET/CAT Concatenation of DataSet objects.
%  Generalized concatenation of DataSet objects
%  [a; b] is the vertical concatenation of DataSet objects
%  (a) and (b). Any number of DataSet objects can be
%  concatenated within the brackets. For this operation
%  to be defined the following must be true:
%    1) All inputs must be valid dataset objects.
%    2) The Dataset 'type' fields must all be the same.
%    3) Concatenation is along the dimension dim. The
%       remaining dimension sizes must match.
%
%  This is similar to matrix concatenation, but each field
%  is treated differently. In structure notation this is:
%    z.name      = a.name;
%    z.type      = a.type;
%    z.author    = a.author
%    z.date      date of concatenation
%    z.moddate   date of concatenation
%    z.data      = cat(dim,a.data,b.data,c.data,...);
%    z.label     mode dim label sets are concatenated, and new
%                label sets are created for all other modes
%    z.axisscale mode dim axisscale sets are concatenated, and new
%                axisscale sets are created for all other modes
%    z.imageaxisscale mode dim axisscale sets are concatenated, and new
%                axisscale sets are created for all other modes
%    z.title     new label sets are created
%    z.class     mode dim class sets are concatenated with
%                empty classes set to zeros, and new
%                class sets are created for all other modes
%    z.description concatenates all descriptions
%    z.include   mode dim include sets are concatenated all
%                others are taken from first dataset
%    z.userdata  if more than one input has userdata it is
%                filled into a cell array, if only one input
%                has userdata it is returned, else it is empty
%
%I/O: z = cat(dim,a,b,c);
%
%See also: DATASET, DATASET/HORZCAT, DATASET/VERTCAT

%Copyright Eigenvector Research, Inc. 2001-2019
%jms 11/06/01 generalized from vertcat and horzcat
%jms 11/06/02 (funny, one year later!) fix bug related to having different
%    numbers of sets of labels, classes, or axisscales in each datatset
%  -allowed cat in dim > ndim (e.g. cat of slabs)
%  -history field note of cat operation
%  -attempt cat of datasets and non-datasets by making datasets from nons
%jms 11/07/02
%  -name and author taken from ALL datasets
%jms 4/24/03
%  -fixed various multi-set problems
%  -fixed include with n-way dataset cat problem
%  -renamed "includ" to "include"
%jms 5/8/03
%  -fixed single-slab to multi-slab bug
%jms 7/24/03
%  -fixed single-slab to multi-slab bug associated with titles
%jms 7/24/04
%  -fixed "empty fieldname stored as double instead of char" bug
%  -title name field better handled
%jms 2/16/05
%  -added image support
%rsk 02/16/05
%  -additional image support for imagesize and mode.
%jms 02/17/05
%  -test for duplicate fields before augmenting new sets
%  -allow cat of empty DSOs (has no effect)
%jms 10/31/05 -fixed bug associated with unequal non-empty "description" fields

ndso     = length(varargin);

if nargin<2;
  error('Must supply dim and at least one dataset');
end

for ii=1:ndso
  if ~isa(varargin{ii},'dataset')
    varargin{ii} = dataset(varargin{ii});
    %     error('All inputs must be dataset objects.')
  end
  if strcmpi('batch',varargin{ii}.type)
    error('Not defined for dataset objects of type ''batch''.')
  end
end

%drop all-empty DSOs
drop = [];
for ii=1:ndso
  if isempty(varargin{ii}.data);
    drop = [drop ii];
  end
end
varargin(drop) = [];
ndso = length(varargin);

switch ndso
  case 0
    y = dataset;
    return
  case 1
    y = varargin{1};
    return
end

[mytimestamp,mytimevect] = timestamp;  %note time we're doing this

y         = dataset;
y.name    = varargin{1}.name;

%history
y.history = [varargin{1}.history];
name = [];
for ii=1:ndso;
  thisname = inputname(ii+1);
  if isempty(thisname);
    thisname = ['"' varargin{ii}.name '"'];
  end
  if isempty(thisname);
    thisname = 'unknown_dataset';
  end
  name = [name ',' thisname];
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
notes  = ['   % ' mytimestamp caller ];
y.history = [y.history {['x=cat(' num2str(dim) name ')' notes]}];

for ii=2:ndso;
  if ~isempty(varargin{ii}.name) & ~strcmp(y.name,varargin{ii}.name);
    if isempty(y.name)
      y.name = [varargin{ii}.name];
    elseif ~strcmpi(y.name,varargin{ii}.name)
      y.name = [y.name ',' varargin{ii}.name];
    end
  end
  y.history = [y.history {sprintf('  + CONCAT "%s" (moddate: %s)',varargin{ii}.uniqueid,timestamp(varargin{ii}.moddate))}];
end
y.author  = varargin{1}.author;
for ii=2:ndso;
  if ~isempty(varargin{ii}.author) & ~strcmp(y.author,varargin{ii}.author);
    if isempty(y.author)
      y.author = [varargin{ii}.author];
    elseif ~strcmpi(y.author,varargin{ii}.author)
      y.author = [y.author ',' varargin{ii}.author];
    end
  end
end
y.date    = mytimevect;
y.moddate = mytimevect;

otherdims        = setdiff(1:ndims(varargin{1}.data),dim);

dimsize  = zeros(ndso,1);
for ii=1:ndso
  dimsize(ii) = size(varargin{ii}.data,dim);
end

%handle imagemode/size fields (if necessary)
y.type = varargin{1}.type;
y.imagesize = varargin{1}.imagesize;
y.imagemode = varargin{1}.imagemode;
if strcmp(y.type,'image'); %Test if this is a type image
  %If first is type image, then result is type image if we can keep it that
  %way
  %Reconcile image dimentionality (2D vs 3D, etc)
  maxsv = max(cellfun(@(i) length(i.imagesize),varargin));
  for ii = 1:ndso;
    if strcmpi(varargin{ii}.type,'image')  %image DSO, make sure its got all image dims
      dif = maxsv - length(varargin{ii}.imagesize);
      if dif > 0
        varargin{ii}.imagesize = [varargin{ii}.imagesize ones(1,dif)];
      end
    else  %NON image DSO, enter simple info (row-wise image)
      varargin{ii}.imagemode = y.imagemode;
      varargin{ii}.imagesize = [size(varargin{ii}.data,y.imagemode) ones(1,maxsv-1)];
    end        
  end

  for ii=2:ndso; %Loop over input arguments
    % * They must all have the same imagemode (note: ERROR)
    if strcmpi(varargin{ii}.type,'image') & varargin{ii}.imagemode ~= y.imagemode
      error('Image modes must be the same to concatenate image datasets.');
    end
    %ADD:  copy imagemode (always - no need to modify)
    if dim~=y.imagemode %  if not cat on imagemode
      %     test imagesize of sources:
      %       equal? use original imagesize
      %       notequal? set imagesize=[mx1]
      if strcmpi(varargin{ii}.type,'image') 
        
        if prod(y.imagesize) ~= prod(varargin{ii}.imagesize)
          %No way to reconcile data sizes.
          error('Total number of pixels is not equal for images to concatenate. Check sizes.');
        end
        
        if length(y.imagesize)~=length(varargin{ii}.imagesize) | any(varargin{ii}.imagesize ~= y.imagesize)
          y.imagesize = [prod(y.imagesize) 1];
        end
        
      end

    else %  if cat on imagemode
      %     test imagesize EXCEPT last dim (e.g. 2-way image, columns don't have
      %     to match; 3-way image, slabs don't have to match)
      %     if necessary sizes match, use original EXCEPT last which is sum of
      %        the concatenated items
      %     if necessary sizes do NOT match, imagesize=[mx1]
      
      if length(y.imagesize) == length(varargin{ii}.imagesize) & varargin{ii}.imagesize(1:end-1) == y.imagesize(1:end-1)
        y.imagesize(end) = y.imagesize(end) + varargin{ii}.imagesize(end);
      else
        y.imagesize = [(prod(y.imagesize)+prod(varargin{ii}.imagesize)) 1];
      end
    end
  end
end

%concatenate the data
for ii=1:ndso;
  s1{ii} = varargin{ii}.data;
  empty(ii) = isempty(s1{ii});
end
s1 = s1(~empty);  %cull empty data sets
%test for size mismatch
for ii=1:length(s1);  
  if ii > 1;
    if ndims(s1{ii}) ~= ndims(s1{1}) & ~(abs(ndims(s1{ii})-ndims(s1{1}))==1 & dim==max([ndims(s1{ii}) ndims(s1{1})]))  ;
      %if # of dims don't match AND this isn't the special case of single-slab concatenated to multiple-slabs
      error('Number of dimensions in each data sets must match');
    end
    for ij = otherdims;
      if size(s1{ii},ij) ~= size(s1{1},ij) & size(s1{ii},ij)>0;
        error('All data sets must match in size except dim %i',dim);
      end
    end
  end
end
y.data = cat(dim,s1{:});

%label 
y.label = varargin{1}.label;
% all modes except dim
for ii=2:ndso   %loop over number of input arguments
  ia = size(y.label,3); %number of label sets
  for ik=1:size(varargin{ii}.label,3)
    i1 = false;  
    for ij=otherdims   %loop over number of dimensions
      if ~isempty(varargin{ii}.label{ij,1,ik})
        i1 = true;
        break
      end
    end
    if i1
      anyused = false; %will only be true if we actually added a set
      for ij=otherdims
        %does this set match any currently in field now?
        newset = varargin{ii}.label{ij,1,ik};
        if ~isempty(newset); 
          use = true; 
          for jk=1:ia
            oneset = y.label{ij,1,jk};
            if ndims(oneset)==ndims(newset) & all(size(oneset)==size(newset)) & all(oneset==newset)
              use = false;
              break
            end
          end
          if use %if not, use it
            y.label{ij,1,ia+1} = varargin{ii}.label{ij,1,ik};
            y.label{ij,2,ia+1} = varargin{ii}.label{ij,2,ik};
            anyused = true;
          end
        end
      end
      if anyused
        ia = ia + 1;
      end
    end
  end
end
% mode dim
ia   = size(varargin{1}.label,3);
for ii=2:ndso   %loop over the number of input arguments
  ia = max([ia size(varargin{ii}.label,3)]);
end
for ii=1:ia   %loop over sets
  i1 = false;
  for ij=1:ndso
    if size(varargin{ij}.label,3)>=ii & size(varargin{ij}.label,1)>=dim & ~isempty(varargin{ij}.label{dim,1,ii})
      i1 = true;
      break
    end
  end
  if i1
    s1 = '';
    s2 = ' ';
    name = [];
    for ij=1:ndso
      if size(varargin{ij}.label,3)<ii | size(varargin{ij}.label,1)<dim | isempty(varargin{ij}.label{dim,1,ii})
        s1 = char(s1,s2(ones(dimsize(ij),1),:));
      else
        s1 = char(s1,varargin{ij}.label{dim,1,ii});
        thisname = varargin{ij}.label{dim,2,ii};
        if isempty(name);
          name = thisname;
        elseif ~isempty(thisname) & ~strcmpi(deblank(thisname),deblank(name));
          name = [name ',' thisname];
        end
      end
    end
    y.label{dim,1,ii} = char(deblank(s1(2:end,:)));
    y.label{dim,2,ii} = char(deblank(name));
  else
    y.label{dim,1,ii} = '';
    y.label{dim,2,ii} = '';
  end
end
%make all _____name entries character
for set = 1:size(y.label,3);
  for mode = 1:size(y.label,1);
    y.label{mode,2,set} = char(y.label{mode,2,set});
  end
end

%axisscale/axistype
y.axisscale = varargin{1}.axisscale;
y.axistype = varargin{1}.axistype;
% all modes except dim
for ii=2:ndso   %loop over number of input arguments
  ia = size(y.axisscale,3); %number of axisscale sets
  for ik=1:size(varargin{ii}.axisscale,3)%loop over dso(ii) sets.
    i1 = false;  
    for ij=otherdims   %loop over number of dimensions
      if ~isempty(varargin{ii}.axisscale{ij,1,ik})
        i1 = true;
        break
      end
    end
    if i1
      anyused = false; %will only be true if we actually added a set
      for ij=otherdims
        %does this set match any currently in field now?
        newset = varargin{ii}.axisscale{ij,1,ik};
        if ~isempty(newset);
          use = true;
          for jk=1:ia
            oneset = y.axisscale{ij,1,jk};
            if ndims(oneset)==ndims(newset) & all(size(oneset)==size(newset)) & all(oneset==newset)
              use = false;
              break
            end
          end
          if use %if not, use it
            y.axisscale{ij,1,ia+1} = varargin{ii}.axisscale{ij,1,ik};
            y.axisscale{ij,2,ia+1} = varargin{ii}.axisscale{ij,2,ik};
            y.axistype{ij,ia+1}    = varargin{ii}.axistype{ij,ik};
            anyused = true;
          end
        end
      end
      if anyused
        ia = ia + 1;
      end
    end
  end
end
% mode dim
ia   = size(varargin{1}.axisscale,3);
for ii=2:ndso   %loop over the number of input arguments
  %Find max number of sets for any input dso.
  ia = max([ia size(varargin{ii}.axisscale,3)]);
end
for ii=1:ia   %loop over sets
  i1 = true;    %we WILL concatenate unless any input argument has an empty entry for this set
  for ij=1:ndso
    if size(varargin{ij}.axisscale,3)<ii | size(varargin{ij}.axisscale,1)<dim | isempty(varargin{ij}.axisscale{dim,1,ii})
      i1 = false;
      break
    end
  end
  if i1
    s1 = [];
    name = [];
    atype = [];
    for ij=1:ndso
      s1 = [s1 varargin{ij}.axisscale{dim,1,ii}];
      thisname = varargin{ij}.axisscale{dim,2,ii};
      atype = [atype varargin{ij}.axistype(dim,ii)];
      if isempty(name)
        name = thisname;
      elseif ~isempty(thisname) & ~strcmpi(thisname,name);
        name = [name ',' thisname];
      end
    end
    %Get axis type from precedence of 1-discrete 2-stick 3-continuous 4-none
    plist = {'none' 'continuous' 'stick' 'discrete'};
    [junk ploc] = ismember(atype,plist);
    pval = max(ploc);
    pval = plist{pval};
    
    y.axisscale{dim,1,ii} = s1;
    y.axisscale{dim,2,ii} = char(deblank(name));
    y.axistype{dim,ii}    = pval; %Use first DSO axistype.
  else
    y.axisscale{dim,1,ii} = [];
    y.axisscale{dim,2,ii} = '';
    y.axistype{dim,ii}    = 'none';
  end
end
%make all _____name entries character
for set = 1:size(y.axisscale,3);
  for mode = 1:size(y.axisscale,1);
    y.axisscale{mode,2,set} = char(y.axisscale{mode,2,set});
    %Set all empty axistype fields to 'none'.
    if isempty(y.axistype{mode,set})
      y.axistype{mode,set} = 'none';
    end
  end
end

%imageaxisscale/imageaxistype
y.imageaxisscale = varargin{1}.imageaxisscale;
y.imageaxistype = varargin{1}.imageaxistype;
% all modes except dim
for ii=2:ndso   %loop over number of input arguments
  ia = size(y.imageaxisscale,3); %number of imageaxisscale sets
  for ik=1:size(varargin{ii}.imageaxisscale,3)%loop over dso(ii) sets.
    i1 = false;  
    for ij=otherdims   %loop over number of dimensions
      if ~isempty(varargin{ii}.imageaxisscale{ij,1,ik})
        i1 = true;
        break
      end
    end
    if i1
      anyused = false; %will only be true if we actually added a set
      for ij=otherdims
        %does this set match any currently in field now?
        newset = varargin{ii}.imageaxisscale{ij,1,ik};
        if ~isempty(newset);
          use = true;
          for jk=1:ia
            oneset = y.imageaxisscale{ij,1,jk};
            if ndims(oneset)==ndims(newset) & all(size(oneset)==size(newset)) & all(oneset==newset)
              use = false;
              break
            end
          end
          if use %if not, use it
            y.imageaxisscale{ij,1,ia+1} = varargin{ii}.imageaxisscale{ij,1,ik};
            y.imageaxisscale{ij,2,ia+1} = varargin{ii}.imageaxisscale{ij,2,ik};
            y.imageaxistype{ij,ia+1}    = varargin{ii}.imageaxistype{ij,ik};
            anyused = true;
          end
        end
      end
      if anyused
        ia = ia + 1;
      end
    end
  end
end
% mode dim
ia   = size(varargin{1}.imageaxisscale,3);
for ii=2:ndso   %loop over the number of input arguments
  %Find max number of sets for any input dso.
  ia = max([ia size(varargin{ii}.imageaxisscale,3)]);
end
for ii=1:ia   %loop over sets
  i1 = true;    %we WILL concatenate unless any input argument has an empty entry for this set
  for ij=1:ndso
    if size(varargin{ij}.imageaxisscale,3)<ii | size(varargin{ij}.imageaxisscale,1)<dim | isempty(varargin{ij}.imageaxisscale{dim,1,ii})
      i1 = false;
      break
    end
  end
  if i1
    s1 = [];
    name = [];
    atype = [];
    for ij=1:ndso
      s1 = [s1 varargin{ij}.imageaxisscale{dim,1,ii}];
      thisname = varargin{ij}.imageaxisscale{dim,2,ii};
      atype = [atype varargin{ij}.imageaxistype(dim,ii)];
      if isempty(name)
        name = thisname;
      elseif ~isempty(thisname) & ~strcmpi(thisname,name);
        name = [name ',' thisname];
      end
    end
    %Get imageaxis type from precedence of 1-discrete 2-stick 3-continuous 4-none
    plist = {'none' 'continuous' 'stick' 'discrete'};
    [junk ploc] = ismember(atype,plist);
    pval = max(ploc);
    pval = plist{pval};
    
    y.imageaxisscale{dim,1,ii} = s1;
    y.imageaxisscale{dim,2,ii} = char(deblank(name));
    y.imageaxistype{dim,ii}    = pval; %Use first DSO imageaxistype.
  else
    y.imageaxisscale{dim,1,ii} = [];
    y.imageaxisscale{dim,2,ii} = '';
    y.imageaxistype{dim,ii}    = 'none';
  end
end
%make all _____name entries character
for set = 1:size(y.imageaxisscale,3);
  for mode = 1:size(y.imageaxisscale,1);
    y.imageaxisscale{mode,2,set} = char(y.imageaxisscale{mode,2,set});
    %Set all empty imageaxistype fields to 'none'.
    if isempty(y.imageaxistype{mode,set})
      y.imageaxistype{mode,set} = 'none';
    end
  end
end

%title
y.title = varargin{1}.title;
% all modes except dim
for ii=2:ndso   %loop over number of input arguments
  ia = size(y.title,3); %number of label sets
  for ik=1:size(varargin{ii}.title,3)
    anyused = false; %will only be true if we actually added a set
    for ij=otherdims
      %does this set match any currently in field now?
      newset = varargin{ii}.title{ij,1,ik};
      if ~isempty(newset);
        use = true;
        for jk=1:ia
          oneset = y.title{ij,1,jk};
          if strcmpi(deblank(oneset),deblank(newset))
            use = false;
            break
          end
        end
        if use %if not, use it
          y.title{ij,1,ia+1} = varargin{ii}.title{ij,1,ik};
          y.title{ij,2,ia+1} = varargin{ii}.title{ij,2,ik};
          anyused = true;
        end
      end
    end
    if anyused
      ia = ia + 1;
    end
  end
end
% mode dim
ia   = size(varargin{1}.title,3);
for ii=2:ndso   %loop over the number of input arguments
  ia = max([ia size(varargin{ii}.title,3)]);
end
for ii=1:ia   %loop over sets
  i1 = false;
  for ij=1:ndso
    if size(varargin{ij}.title,3)>=ii & size(varargin{ij}.title,1)>=dim & ~isempty(varargin{ij}.title{dim,1,ii})
      i1 = true;
      break
    end
  end
  if i1
    ntitle = '';
    name  = '';
    for ij=1:ndso
      if size(varargin{ij}.title,3)<ii | size(varargin{ij}.title,1)<dim | isempty(varargin{ij}.title{dim,1,ii})
        %nothing to do...
      else
        thistitle       = varargin{ij}.title{dim,1,ii};
        if isempty(ntitle)
          ntitle = thistitle;
        elseif ~isempty(thistitle) & ~strcmpi(ntitle,thistitle)
          ntitle = [ntitle ',' thistitle];
        end
        thisname = varargin{ij}.title{dim,2,ii};
        if isempty(name)
          name = thisname;
        elseif ~isempty(thisname) & ~strcmpi(thisname,name);
          name = [name ',' thisname];
        end
      end
    end
    y.title{dim,1,ii} = char(deblank(ntitle));
    y.title{dim,2,ii} = char(deblank(name));
  else
    y.title{dim,1,ii} = '';
    y.title{dim,2,ii} = '';
  end
end
%make all _____name entries character
for set = 1:size(y.title,3);
  for mode = 1:size(y.title,1);
    y.title{mode,2,set} = char(y.title{mode,2,set});
  end
end

%classlookup
%Do classlookup BEFORE class so new numeric classes can be assigned if
%conflicting string classes are found.
y.classlookup = varargin{1}.classlookup;
% all modes except dim
% Append lookup tables as new sets to match class sets created below.
for ii=2:ndso   %loop over number of input arguments
  ia = size(y.classlookup,2); %number of class sets
  for ik=1:size(varargin{ii}.classlookup,2) %loop over number of sets.
    anyused = false; %will only be true if we actually added a set
    for ij=otherdims
      %does this set match any currently in field now?
      newset = varargin{ii}.classlookup{ij,ik};
      if ~isempty(newset);
        use = true;
        for jk=1:ia %Loop through existing sets.
          oneset = y.classlookup{ij,jk};%Does this class already exist.
          
          %Do the sizes of the classlookup field match, are all elements identical.
          
          %If one class is empty and the other not, treat as non duplicate,
          %this can happen and caused an error in DSE for Bob. Class lookup
          %not getting getting copied. After cat lu table missing set:
          % >> spud.class
          % 
          % ans = 
          % 
          %     [1x18 double]    [1x18 double]                 []
          %                []               []    [1x2047 double]
          % 
          % >> spud.classlookup
          % 
          % ans = 
          % 
          %     {10x2 cell}    {4x2 cell}
          %     { 4x2 cell}            {}

          
          classisempty = false;
          if size(varargin{1}.class,2)>=ik
            if isempty(varargin{1}.class{ij,ik})
              classisempty = true;
            end
            
            if classisempty & isempty(varargin{ii}.class{ij,ik})
              %Both empty.
              classisempty = false;
            end
          end
          
          if isequal(oneset,newset) & ~classisempty
            use = false;%Set matches completely, don't uses.
            break
          end
        end
        if use %if not, use it
          y.classlookup{ij,ia+1} = varargin{ii}.classlookup{ij,ik};
          anyused = true;
        end
      end
    end
    if anyused
      ia = ia + 1;
    end
  end
end
% mode dim
ia   = size(varargin{1}.classlookup,2);
for ii=2:ndso   %loop over the number of input arguments
  %Find largest number of sets in any input dso.
  ia = max([ia size(varargin{ii}.classlookup,2)]);
end
for ii=1:ia %loop over sets
  i1 = false;
  for ij=1:ndso %loop over dso
    %Make sure the dso being concatenated has enough sets, has a lookup
    %table in the dimension being concatenated, and that the table isn't
    %empty.
    if size(varargin{ij}.classlookup,2)>=ii ... %Total number of sets in ij dso greater than current set number(ii). 
       & size(varargin{ij}.classlookup,1)>=dim ... %Total number of dims in ij dso >= concat dimension.
       & ~isempty(varargin{ij}.classlookup{dim,ii}) %Lookup table is not empty in set ii at concat dimention. 
      i1 = true;
      break
    end
  end
  if i1
    s1 = [];
    for ij=1:ndso
      %For each dso at the 'ii' set merge the lookup table.
      % Check to see if there's actually something there to merge.
      if size(varargin{ij}.classlookup,2)>=ii... %
         && size(varargin{ij}.classlookup,1)>=dim...
         && ~isempty(varargin{ij}.classlookup{dim,ii})
        if isempty(s1)
          %Get existing table. This will enact precedence of class
          %assignmets where the first DSO has precedence.
          s1 = varargin{ij}.classlookup{dim,ii};
        else
          %Merge existing tables.
          curtbl = varargin{ij}.classlookup{dim,ii};
          [s1 newcls]= tablemerge(s1,curtbl);
          if ~isempty(newcls)
            %Need to use NaN so that old values don't confuse reassignment,
            %e.g., class 'aaa' is 1 in first table and 3 in second table
            %then second table 3s need to be converted to 1s.
            newclsvec = nan(1,size(varargin{ij}.class{dim,1,ii},2));
            for jj = 1:length(newcls)
              %Update merging DSO with new class info in needed.
              newclsvec(varargin{ij}.class{dim,1,ii}==newcls(jj).oldnum) = newcls(jj).newnum;
            end
            %Add unchanged values.
            myloc = isnan(newclsvec);
            newclsvec(myloc) = varargin{ij}.class{dim,1,ii}(myloc);
            %Put back into dataset.
            varargin{ij}.class{dim,1,ii} = newclsvec;
          end
        end
      else
        %If a DSO doesn't have an existing class then the class code above
        %will make all values zero so add an entry to the lookup table for
        %that case if it doesn't exist.
        if isempty(s1)
          s1 = {0 'Class 0'};
        else
          zpos = find(ismember([s1{:,1}],0));
          if isempty(zpos)
            s1 = [{0 'Class 0'};s1];
          end
        end
      end
    end
    y.classlookup{dim,ii} = s1;
    %y.class{dim,2,ii} = char(deblank(name));
  else
    y.classlookup{dim,ii} = {};
    %y.class{dim,2,ii} = '';
  end
end

%class
y.class = varargin{1}.class;
% all modes except dim
for ii=2:ndso   %loop over number of input arguments
  ia = size(y.class,3); %number of class sets
  for ik=1:size(varargin{ii}.class,3)
    anyused = false; %will only be true if we actually added a set
    for ij=otherdims
      %does this set match any currently in field now?
      newset = varargin{ii}.class{ij,1,ik};
      if ~isempty(newset);
        use = true;
        for jk=1:ia %Loop through existing sets.
          oneset = y.class{ij,1,jk};%Does this class already exist.
          if ndims(oneset)==ndims(newset) & all(size(oneset)==size(newset)) & all(oneset==newset)%Do the sizes of the class field match, are all elements identical.
            use = false;%Set matches completely, don't uses.
            %Check classid, if match then don't use otherwise make new set
            %and copy classid.
            
            break
          end
        end
        if use %if not, use it
          y.class{ij,1,ia+1} = varargin{ii}.class{ij,1,ik};
          y.class{ij,2,ia+1} = varargin{ii}.class{ij,2,ik};
          anyused = true;
        end
      end
    end
    if anyused
      ia = ia + 1;
    end
  end
end
% mode dim
ia   = size(varargin{1}.class,3);
for ii=2:ndso   %loop over the number of input arguments
  ia = max([ia size(varargin{ii}.class,3)]);
end
for ii=1:ia   %loop over sets
  i1 = false;
  for ij=1:ndso
    if size(varargin{ij}.class,3)>=ii & size(varargin{ij}.class,1)>=dim & ~isempty(varargin{ij}.class{dim,1,ii})
      i1 = true;
      break
    end
  end
  if i1
    s1 = [];
    name = [];
    for ij=1:ndso
      if size(varargin{ij}.class,3)<ii | size(varargin{ij}.class,1)<dim | isempty(varargin{ij}.class{dim,1,ii})
        s1 = [s1, zeros(1,dimsize(ij))];
      else
        s1 = [s1, varargin{ij}.class{dim,1,ii}];%Need to check tables.
        thisname = varargin{ij}.class{dim,2,ii};
        if isempty(name)
          name = thisname;
        elseif ~isempty(thisname) & ~strcmpi(thisname,name);
          name = [name ',' thisname];%Use this for classid too.
        end
      end
    end
    y.class{dim,1,ii} = s1;
    y.class{dim,2,ii} = char(deblank(name));
  else
    y.class{dim,1,ii} = [];
    y.class{dim,2,ii} = '';
  end
end
%make all _____name entries character
for set = 1:size(y.class,3);
  for mode = 1:size(y.class,1);
    y.class{mode,2,set} = char(y.class{mode,2,set});
  end
end

%include
y.include = varargin{1}.include;
if 0;     %the next part of the code is disabled
  % all modes except dim
  for ii=2:ndso   %loop over number of input arguments
    ia = size(y.include,3); %number of sets
    for ik=1:size(varargin{ii}.include,3)
      i1 = false;  
      for ij=otherdims   %loop over number of dimensions
        if ~isempty(varargin{ii}.include{ij,1,ik})
          i1 = true;
          break
        end
      end
      if i1
        for ij=otherdims
          y.include{ij,1,ia+1} = varargin{ii}.include{ij,1,ik};
          y.include{ij,2,ia+1} = varargin{ii}.include{ij,2,ik};
        end
        ia = ia+1;
      end
    end
  end
end
%mode dim
ia   = size(varargin{1}.include,3);
for ii=2:ndso   %loop over the number of input arguments
  ia = max([ia size(varargin{ii}.include,3)]);
end
for ii=1:ia   %loop over sets
  if size(varargin{1}.include,1)>=dim;
    s1 = varargin{1}.include{dim,1,ii};
  else
    s1 = 1;
  end
  for ij=2:ndso
    if size(varargin{ij}.include,3) < ii | size(varargin{ij}.include,1)<dim
      s1 = [s1, sum(dimsize(1:(ij-1)))+1:sum(dimsize(1:ij))];  %include ALL if no include set
    elseif ~isempty(varargin{ij}.include{dim,1,ii});
      s1 = [s1, sum(dimsize(1:(ij-1)))+varargin{ij}.include{dim,1,ii}];
    end
  end
  y.include{dim,1,ii} = s1;
end

%Note: on 8/30/00 the following code was used, but failed
%to provide an inputname (should be checked with R12 to
%see if the same problem exists). The Code following the %---
%is used until this is fixed by the MathWorks
%description
%for ii=1:ndso
%  if isempty(varargin{ii}.description)
%    y.description = strvcat(y.description,[inputname(ii),':']);
%  else
%    y.description = strvcat(y.description,[inputname(ii),':'], ...
%                            varargin{ii}.description);
%  end
%end
%---
for ii=1:ndso
  if prod(size(y.description))~= prod(size(varargin{ii}.description)) | (any(y.description(:)~=varargin{ii}.description(:)));
    y.description = strvcat(y.description,varargin{ii}.description);
  end
end

%userdata
i1    = 0;
i2    = [];
for ii=1:ndso
  if ~isempty(varargin{ii}.userdata)
    i1 = i1+1;
    i2 = [i2 ii];
  end
end
if i1>1
  y.userdata = cell(i1,1);
  for ii=1:i1
    y.userdata{ii,1} = varargin{i2(ii)}.userdata;
  end
elseif i1==1
  y.userdata = varargin{i2}.userdata;
end


