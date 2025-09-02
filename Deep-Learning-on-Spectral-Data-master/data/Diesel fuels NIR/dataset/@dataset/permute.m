function out = permute(in,order)
%DATASET/PERMUTE Permute array dimensions of a Dataset Object.
%  Rearranges the dimensions of A so that they are in the order specified
%  by the vector ORDER.  The dataset produced has the same values of A but
%  the order of the subscripts needed to access any particular element are
%  rearranged as specified by ORDER. The elements of ORDER must be a
%  rearrangement of the numbers from 1 to N.
%  All informational fields are also reordered as necessary.
%
%  PERMUTE and ipermute are a generalization of transpose (.') for N-D
%  arrays.
%
%I/O: b = permute(a,order)
%
%See also: IPERMUTE

%Copyright Eigenvector Research, Inc. 2003-2019
%jms 4/24/03 -renamed "includ" to "include"

if nargin<2;
  error('input ORDER is missing');
end

if length(order) < length(size(in));
  error('ORDER must have at least N elements for an N-D array.')
end

if any(order<1) | length(order)<max(order)
  error('ORDER contains an invalid permutation index.');
end

inputorder = order;
out = dataset(permute(in.data,order));

if max(order)>ndims(out.data);
  order(ndims(out.data)+1:end) = [];
end

%copy these as-is
out.name      = in.name;
out.type      = in.type;
out.imagesize = in.imagesize;
out.author    = in.author;
out.date      = in.date;

%reorder first dim of add info cells
mdim = max(order);

%test and expand these fields so that permute will work (if ndims is going up)
if size(in.label,1)<mdim; in.label(mdim,end,end)={[]}; end
if size(in.axisscale,1)<mdim; in.axisscale(mdim,end,end)={[]}; end
if size(in.title,1)<mdim; in.title(mdim,end,end)={[]}; end
if size(in.class,1)<mdim; in.class(mdim,end,end)={[]}; end
if size(in.include,1)<mdim; in.include(mdim,end,end)={[]}; end
if size(in.axistype,1)<mdim; in.axistype(mdim,end)={'none'}; end
if size(in.classlookup,1)<mdim; in.classlookup(mdim,end)={{}}; end

%Assign new position of imagemode.
if strcmp(in.type,'image')
  out.imagemode = find(ismember(order,in.imagemode));
end

%copy value into output field of same name
out.label       = in.label(order,:,:);
out.axisscale   = in.axisscale(order,:,:);
out.axistype    = in.axistype(order,:);
out.title       = in.title(order,:,:);
out.class       = in.class(order,:,:);
out.classlookup = in.classlookup(order,:);
out.include     = in.include(order,:,:);

if strcmpi(in.type,'image')
  %Relocate image mode to new position.
  out.imagemode = find(ismember(order,in.imagemode));
  
  %copy these as-is
  out.imageaxisscale   = in.imageaxisscale;
  out.imageaxistype    = in.imageaxistype;
end

%copy these as-is
out.description = in.description;
out.history     = in.history;
out.userdata    = in.userdata;

%put entry into history field
z = inputname(1);
if isempty(z); z = 'ans'; end
[mytimestamp,out.moddate] = timestamp;   %and update moddate
notes  = ['   % ' mytimestamp];
out.history(end) = {[z ' = ' z ''';' notes]};
out.history(end+1) = {[z ' = permute(' z ',[' num2str(inputorder) ']);' notes]};

