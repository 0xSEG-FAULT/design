function out = reshape(in,varargin)
%DATASET/RESHAPE Change size of a DataSet object.
% RESHAPE(X,M,N) returns the M-by-N matrix whose elements are taken
% columnwise from X.  An error results if X does not have M*N elements.
% 
% RESHAPE(X,M,N,P,...) or RESHAPE(X,[M N P ...]) returns an N-D array with
% the same elements as X but reshaped to have the size M-by-N-by-P-by-...
% Note that M*N*P*... must be the same as PROD(SIZE(X)).
%
% This function generally follows the I/O of the standard reshape command
% but some options may not be available here.
%
%I/O: rx = reshape(x,a,b,c,...)
%I/O: rx = reshape(x,[a b c])

%Copyright Eigenvector Research, Inc. 2003-2019

if nargin>2
  newsz = [varargin{:}];
elseif nargin==2
  newsz = varargin{1};
else
  error('Not enough input arguments.');
end

sz = size(in);
if length(sz)==length(newsz) & all(sz==newsz)
  %in is already the given size, return as-is
  out = in;
  return
end

if prod(sz)~=prod(newsz)
  error('To RESHAPE the number of elements must not change.')
end

%Do resize
out = dataset(reshape(in.data,newsz));

%try copying labels
copied = false(1,length(sz));

%look at initial mode(s)
n = min(length(sz),length(newsz));
issame = sz(1:n)==newsz(1:n);
issame(min(find(~issame)):end) = 0;  %ignore any matching modes after first mismatch
for from=find(issame)
  if ~copied(from)
    out = copyfields(in,out,from);
    copied(from) = true;
  end
end

%look at last mode(s)
issame = sz(end:-1:end-n+1)==newsz(end:-1:end-n+1);
issame(min(find(~issame)):end) = 0;  %ignore any matching modes after first mismatch
for j=find(issame)
  from = ndims(in)-j+1;
  to   = ndims(out)-j+1;
  if ~copied(from)
    out = copyfields(in,out,from,to);
    copied(from) = true;
  end  
end
  
%copy these as-is
out.name      = in.name;
out.type      = in.type;
out.author    = in.author;
out.date      = in.date;

%if was image and we copied the image mode as-is, copy image size and imagemode
if strcmp(in.type,'image')
  if copied(in.imagemode)
    out.imagesize = in.imagesize;
    out.imagemode = in.imagemode;
    out.imageaxisscale = in.imageaxisscale;
    out.imageaxistype = in.imageaxisscale;
  else
    %we broke up the imagemode, change to data
    out.type = 'data';
  end
end

%modify mod date
out.moddate = clock;

out.description = in.description;
out.userdata    = in.userdata;

%assign history
out.history     = in.history;
caller = '';
try
  [ST,I] = dbstack;
  if length(ST)>1;
    [a,b,c]=fileparts(ST(2).name); 
    caller = ['[' b c ']'];
  end
catch
end
if ~isempty(caller)
  notes  = [' - ' caller ];
else
  notes = '';
end

objname = inputname(1);

time = clock;
tstamp = [ ' % ' datestr(time,'dd-mmm-yyyy') ' ' datestr(time,'HH:MM') sprintf(':%06.3f',time(end))];

if isempty(out.history{1})
  ihis   = 1;
else
  ihis   = length(out.history)+1;
end
out.history{ihis} = ['reshape(' objname ',[ ' sprintf('%i ',newsz) '])' tstamp notes];


%=====================================================
function out = copyfields(in,out,from,to,field)

if nargin<4
  to = from;
end
if nargin<5;
  %no field, copy all (using recursive call)
  for fieldname = {'class' 'classlookup' 'axisscale' 'imageaxisscale' 'axistype' 'imageaxistype' 'label' 'title' 'include'};
    out = copyfields(in,out,from,to,fieldname{:});
  end
else
  %got a field - copy it
  temp = in.(field)(from,:,:);
  sz = size(temp);
  inds = cell(1,length(sz));
  inds{1} = to;
  for j=2:length(sz)
    inds{j} = 1:sz(j);
  end
  out.(field)(inds{:}) = temp;
end

