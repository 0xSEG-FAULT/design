function value =get(x,feyld,vdim,vset)
%DATASET/GET Get field (property) values from a DataSet object.
%I/O: value = get(x,'field');
%  when (x) is a DataSet object (See DATASET) this returns
%  the value (value) of the DataSet object field ('field').
%  This syntax is used for the following fields:
%    name         : value is a char vector.
%    author       : value is a char vector.
%    date         : value is a 6 element vector (see CLOCK).
%    moddate      : value is a 6 element vector (last modified date).
%    type         : value is either 'data' or 'image'.
%    data         : value is a double array.
%    userdata     : user defined.
%    description  : value is a char array.
%    history      : cell array of char (e.g. char(get(x,'history)))
%    datasetversion: dataset object version.
%
%I/O: value = get(x,'field',vdim);
%    include      : value is row vector of indices for mode vdim
%I/O: value = get(x,'field',vdim,vset);
%  when (x) is a DataSet object this returns the value (value)
%  for the field ('field') for the specified dimension/mode
%  (vdim) and optional set (vset) {default: vset=1}. E.g. (vset) is
%  used when multiple sets of labels are present in (x).
%  This syntax is used for the following fields:
%    label        : value is a char array with size(x.data,vdim) rows.
%    labelname    : value is a char row vector.
%    axisscale    : value is a row vector with size(x.data,vdim) real elements.
%    axisscalename: value is a char row vector.
%    title        : value is a char row vector.
%    titlename    : value is a char row vector.
%    class        : value is a row vector with size(x.data,vdim) integer elements.
%    classname    : value is a char row vector.
%
%See Also: DATASET, DATASETDEMO, ISA, DATASET/SET, DATASET/SUBSREF, DATASET/EXPLODE

%Copyright Eigenvector Research, Inc. 2000-2019
%nbg 8/3/00, 8/16/00, 8/17/00, 8/30/00, 10/05/00, 10/10/00
%jms 5/31/01 - revised as call to dataset/subsref
%jms 4/24/03 -renamed "includ" to "include"

%I/O: value = get(x,'field',vdim)
%    include      : value is a row vector of indices for vdim.

switch nargin
  case 1
    value = x;
    return
  case 2
    indxstr = substruct('.',feyld);
  case 3
    indxstr = substruct('.',feyld,'{}',{vdim});
  case 4
    indxstr = substruct('.',feyld,'{}',{vdim vset});
  otherwise
    error('GET requires 2 to 4 inputs.')
end;

try
  value = subsref(x,indxstr);
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

return
