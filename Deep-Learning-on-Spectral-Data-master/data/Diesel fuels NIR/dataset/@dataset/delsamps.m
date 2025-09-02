function x = delsamps(x,inds,vdim,flag)
%DATASET/DELSAMPS Marks/Deletes samples from dataset objects.
%  DELSAMPS is used to mark rows or columns to be
%  "excluded" but still be retaind in a DataSet (i.e.
%  soft delete). It can also be used to permanently
%  remove rows or columns from a DataSet object (i.e.
%  hard delete).
%  Inputs are the original DataSet object (data) and
%  the indices to mark (inds).
%  Optional input (vdim) is the mode/dimension to mark
%    1=rows {default}, and
%    2=columns, etc.
%  Optional input (flag) indicates to mark/soft delete
%  when set to 1 {default}, hard delete when set to 2,
%  or hard "keep"/reorder when set to 3 (when flag is 3,
%  inds are indices to keep).
%  The output is the edited DataSet object (eddata).
%
%I/O: eddata = delsamps(data,inds,vdim,flag)

%Copyright Eigenvector Research, Inc. 2000-2019
%nbg 10/11/00
%jms 4/20/01
%  - fixed parentesis bug (there was a "}" where there should have been a ")" )
%  - fixed index bug (e.g. if includ = 6:10 and user
%      asked to remove 8, it wouldn't remove anything BUT
%      if user asked to remove 2, 7 would be removed)
%jms 5/9/01 - adapted for use with data being other classes (single, uint8, etc).
%nbg 5/11/01 - terminated line 67 x.includ{vdim} = setdiff(x.includ{vdim},inds) with ;
%nbg 5/14/01 - added "hard delete" flag==2
%jms 9/9/02 -added "hard keep/reorder" option (flag==3)
%   -make history notation for both soft AND hard deletions
%jms 4/24/03 -renamed "includ" to "include"
%jms 7/24/03 -fixed bug when multiple titles were present
% jms 11/7/03 -base vdim tests on ndims(x) code instead of ndims(x.data)
%rsk 02/22/2005 -resize image if delete samples in image mode.
%jms 3/14/05 -fixed behavior when all samples or variables deleted

if ~isa(x,'dataset') %redundant test for overloaded methods
  error('Function for DATASET ojects only.')
end
if nargin<2
  error('2 inputs required.')
elseif isempty(x.data)
  warning('EVRI:DSODeleteEmptyData','Dataset data field empty.')
end
if nargin<3
  vdim     = 1;
elseif isempty(vdim)                       %nbg 5/14/01
  vdim     = 1;
elseif isa(x.data,'cell')
  if (vdim>ndims(x.data{1}))|(vdim<1)
    error(['vdim must be integer in [1:',int2str(ndims(x.data{1})),'].'])
  end
else
  if (vdim>ndims(x))|(vdim<1)
    error(['vdim must be integer in [1:',int2str(ndims(x)),'].'])
  end
end
if nargin<4
  flag     = 1;
elseif isempty(find(flag==[1 2 3]))
  error('Unknown option for ''flag''.')
end

if isa(x.data,'cell')
  nsize    = size(x.data{1});
  nsize(1) = length(x.data);
else
  nsize    = size(x);
end
nsize      = nsize(vdim);

if ~isempty(inds) & any(inds<1)|(max(inds)>nsize)
  error(['For vdim = ',int2str(vdim),', inds must be integers in [1:',int2str(nsize),'].'])
end

if isempty(inputname(2))
  %might try to identify and reproduce the vector here
  indsin   = '?FunctionCall';
else
  indsin   = inputname(2);
end

caller = '';
try
  [ST,I] = dbstack;
  b = [];
  if length(ST)>2;
    [a,b,c]=fileparts(ST(3).name);
  elseif length(ST)>1
    [a,b,c]=fileparts(ST(2).name);
  end
  if ~isempty(b)
    caller = ['% [' b c ']'];
  end 
catch
end
x.history{length(x.history)+1} = ['delsamps(',inputname(1),',',indsin,',', ...
  int2str(vdim),',',int2str(flag),');   ' caller];

if flag==1
  x.include{vdim,1} = setdiff(x.include{vdim,1},inds);
  %find difference between what exists and what we're being asked to remove

  if isempty(x.date)
    x.date          = clock;
    x.moddate       = x.date;
  else
    x.moddate       = clock;
  end
elseif flag==2 | flag==3
  %this is a "hard" delete and corresponds to constructing a new data set
  x.date            = clock;
  x.moddate         = x.date;

  iall              = 1:size(x.data,vdim);
  if flag==2
    iuse            = setdiff(iall,inds);
  else    %flag==3 means "hard keep" - keep indicated inds discarding rest (allows reordering too!)
    iuse            = inds;
  end

  %S describes subscripting.
  S      = [];
  S.type = '()';
  S.subs = cell(1,ndims(x));
  [S.subs{1:end}] = deal(':');   % gives  {':' ':' ':' ...}
  S.subs{vdim}              = iuse;        % insert indices into appropriate dim
  x.data                    = subsref(x.data,S);

  %old code to do above using eval:
  %     s         = cell(2,1); s{1}  = ',';      s{2}  = ':';      s  = s(:,ones(1,ndims(x.data)));
  %     s{2,vdim} = ['iuse']'; s     = char(s)'; s     = s(2:end);
  %     eval(['x.data     = x.data(',s,');'])

  for ii=1:size(x.label,3)
    if ~isempty(x.label{vdim,1,ii})
      x.label{vdim,1,ii} = deblank(x.label{vdim,1,ii}(iuse,:));
      x.label{vdim,2,ii} = deblank(x.label{vdim,2,ii});
    end
  end
  for ii=1:size(x.axisscale,3)
    if ~isempty(x.axisscale{vdim,1,ii})
      x.axisscale{vdim,1,ii} = x.axisscale{vdim,1,ii}(iuse);
      x.axisscale{vdim,2,ii} = x.axisscale{vdim,2,ii};
    end
  end
  for ii=1:size(x.title,3)
    if ~isempty(x.title{vdim,1,ii})
      x.title{vdim,1,ii} = deblank(x.title{vdim,1,ii});
      x.title{vdim,2,ii} = deblank(x.title{vdim,2,ii});
    end
  end
  for ii=1:size(x.class,3)
    if ~isempty(x.class{vdim,1,ii})
      x.class{vdim,1,ii} = x.class{vdim,1,ii}(iuse);
      x.class{vdim,2,ii} = x.class{vdim,2,ii};
    end
  end
  for ii=1:size(x.include,3)
    if ~isempty(x.include{vdim,1,ii})
      y                      = zeros(1,length(iall));
      y(x.include{vdim,1,ii}) = ones(1,length(x.include{vdim,1,ii}));
      %If ind/iuse is not a row vector then include will not be a row
      %vector because of 'find' so check and correct if necessary. This can
      %happen when transposing an image with flipud_img, ind/iuse is a
      %matrix.
      thisinclude             = find(y(iuse));
      if size(thisinclude,1)>1
        %Transpose.
        thisinclude = thisinclude';
      end
      x.include{vdim,1,ii}    = thisinclude;
      x.include{vdim,2,ii}    = x.include{vdim,2,ii};
    end
  end

  %If the data being deleted/marked is in the image mode, then change image
  %size to vector of new length.
  if strcmp(x.type, 'image') & any(x.imagemode == vdim)
    %new length using iuse
    x.imagesize = [prod(size(iuse)) 1];
  end

end

