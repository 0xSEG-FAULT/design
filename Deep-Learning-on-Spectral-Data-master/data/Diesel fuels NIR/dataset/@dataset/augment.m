function data = augment(dim,varargin)
%DATASET/AUGMENT Concatenation of DataSet objects.
%  Does cat then adds a new class for conctenated items. By default the
%  concatenated class set will be named "Augmented Data" and be reused on
%  additional concatenations.
% 
%  If dim = -1 then dialog box appears for what dimension to augment in.
%  The code will look for matching sizes in dimesions of first and second
%  dataset inputs for dimensions that are compatible. If only one found
%  then will not show dialog. Returns empty if user cancels dialog.
%
%  If dim = 'some_string' then function will search for the first occurrence
%  of the 'some_string' in the .classname field and make that the dimension
%  and set for the augmented data class.
%
%I/O: z = augment(dim,a,b,c);
%
%See also: DATASET, DATASET/HORZCAT, DATASET/VERTCAT

%Copyright Eigenvector Research, Inc. 1991-2019
%Licensee shall not re-compile, translate or convert "M-files" contained
% in PLS_Toolbox for use with any software other than MATLAB®, without
% written permission from Eigenvector Research, Inc.

persistent usematchvars

%Orginal data to cat to.
data     = varargin{1};

%Check for class name given as 'dim'.
if ischar(dim)
  classname = dim;
  dim = -1;%Default to dialog if can't find classname.
  for i = 1:ndims(data)
    findcls = find(ismember(squeeze(data.class(i,2,:)),classname));
    if ~isempty(findcls)
      dim = i;
    end
  end
else
  classname = 'Augmented Data';
end


%All other data.
varargin = varargin(2:end);

%If dim is -1 or charater then show dialog and ask what dimension to
%augment on.
if dim==-1
  dim = getaugmentdirection(data,varargin{1});
  if isempty(dim)
    %User cancel.
    data = [];
    return
  end
  switch lower(dim)
    case 'rows'
      dim = 1;
    case 'columns'
      dim = 2;
    case 'slabs'
      dim = 3;
    otherwise
      error(['Unrecognized concatenation method "' dim '".'])
  end
end

%Get augment class position.
if dim<=ndims(data)
  newclasspos = find(ismember(squeeze(data.class(dim,2,:)),classname));
else
  newclasspos = [];
end

if ~isempty(newclasspos)
  %Class already exists, add the new samples next index.
  newclasspos = newclasspos(1);  %Make sure only the FIRST class set matching this name is used.
  newclasslu  = data.classlookup{dim,newclasspos};
  nextval     = max([newclasslu{:,1}])+1;
  %Get existing class vector.
  newclass    = data.class{dim,1,newclasspos};
else
  %Create new class.
  %Get next empty position in class field.
  if dim<=ndims(data)
    newclasspos = min(find(cellfun('isempty',data.class(dim,1,:))));
  else
    newclasspos = 1;
  end
  if isempty(newclasspos)
    newclasspos = size(data.class,3)+1;
  end
  %Start at one.
  nextval = 1;
  %Make class name the name of the data.
  myname  = data.name;
  if isempty(myname)
    %If no name then just use Augmented Data.
    myname = ['Augmented Data ' num2str(nextval)];
  end
  %Create new lookup table.
  newclasslu = {nextval myname};
  %Create class vector for data.
  newclass   = ones(1,size(data,dim));
  nextval = nextval+1;
end

if isempty(usematchvars)
  %check if matchvars exists (if no PLS_Toolbox, we can't use this)
  usematchvars = exist('matchvars','file');
end

if usematchvars & dim==1
  success = true(1,length(varargin));
  try
    data = matchvars([{data} varargin]);
  catch
    joinerr = lasterror;
    warning('EVRI:DSOAugmentError',['Error augmenting data. ' lasterr])
    success = false(1,length(varargin));
  end
else
  %can't use matchvars, try simple join
  for i = 1:length(varargin)
    success(i) = 1;
    %If cat multiple datasets and one does not work just show warning.
    try
      data = cat(dim,data,varargin{i});
    catch
      joinerr = lasterror;
      warning('EVRI:DSOAugmentError',['Error augmenting data. ' lasterr])
      success(i) = 0;
    end
  end
end

if ~any(success)
  rethrow(lasterror)
end

for i = 1:length(varargin)
  if success(i)
    %Add a class for augmented data.
    newclass = [newclass nextval*ones(1,size(varargin{i},dim))];
    myname = varargin{i}.name;
    if ~isempty(myname) & ismember(myname,newclasslu(:,2))
      %if a similar name exists, add (#) to the end until we find a unique
      %version
      rep = 2;      
      while ismember([myname '(' num2str(rep) ')'],newclasslu(:,2))
        rep = rep+1;
      end
      myname = [myname ' (' num2str(rep) ')'];
    end
    if isempty(myname)
      myname = ['Augmented Data ' num2str(nextval)];
    end
    newclasslu = [newclasslu; {nextval myname}];
    nextval = nextval+1;
  end
end

data.class{dim,1,newclasspos} = newclass;%Class.
data.class{dim,2,newclasspos} = classname;%Class set name.
data.classlookup{dim,newclasspos} = newclasslu;%Class lookup.

%-----------------------------------------------------
function mydirection = getaugmentdirection(data,newdata)
%GETAUGMENTDIRECTION Get mode of augmentation.

mydirection = '';

match = getmatchdims(data,newdata);

augopts = {'Rows' 'Columns' 'Slabs'};
match = intersect(1:3,find(match));
augopts = augopts(match);
if length(match)==1;
  method = augopts{:};
else
  if length(augopts)<3;
    augopts{end+1} = 'Cancel';
  end
  if length(augopts)>3;
    augopts = augopts(1:3);
  end
  method = evriquestdlg('Augment in which direction?', ...
    'Augment Data', ...
    augopts{:},augopts{1});
  if strcmp(method,'Cancel')
    return
  end
end

mydirection = method;

%-----------------------------------------------------
function match = getmatchdims(data,newdata)
%GETMATCHDIMS Get matching dims for two size vectors.

szA = size(data); %size(data.data);
szA(end+1) = 1;              %allow for
szB = size(newdata);
szB(end+1:length(szA)) = 1;  %fill in remaining lengths
szA(end+1:length(szB)) = 1;  %fill in remaining lengths
match = szB.*0;     %flag indicating if a given dim can be concated.
for j=1:length(szA);
  dims = setdiff(1:length(szB),j);  %dims which must match
  %enough dims?
  if max(dims)<=length(szA)
    match(j) = all(szA(dims)==szB(dims));
  end
end
match(1) = 1;  %ALWAYS allow rows

%-----------------------------------------------------
function test
%Test casses for augment.

load arch

a = arch;

b = dataset(rand(75,5));
b.name = 'B';
c = b;
c.name = 'C';

d = dataset(rand(20));
d.name = 'D';

e = augment(2,a,b,c);
e.classlookup{2,1}

f = augment(1,e,d);
f.classlookup{1,2}

g = augment(-1,a,a,a);
g.classlookup

%Augment and slabs should give 75x10x3 and 2 warnings.
h = augment(-1,a,a,c,d,a);
h.classlookup{3,1}

a.class{1,3} = ones(75,1);
a.class{2,3} = ones(10,1);
a.classname{1,3} = 'class13';
a.classname{2,3} = 'class23';

j = augment('class13',a,arch);
j = augment('class23',a,b);


