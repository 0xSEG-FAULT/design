function data = dataset(varargin)
%DATASET DataSet object class constructor.
% Creates a DataSet object which can contain data along with related
%  informational fields including: 
%   
%    name           : name of data set.
%    author         : authors name.
%    date           : date of creation.
%    moddate        : date of last modification.
%    type           : either 'data', 'batch' or 'image'.
%    size           : size vector of data.
%    sizestr        : string of size.
%    imagesize      : size of image modes for type 'image' data. Should be
%                     used to unfold/refold image data in .data field.
%    imagesizestr   : string of imagesize.
%    foldedsize     : size of folded image data returned by .imagedataf.
%    foldedsizestr  : string of foldedsize.
%    imagemode      : mode where spatial data has been unfolded to.
%    imageaxisscale : axis scales for each image mode.
%    imageaxisscalename  : descriptive name for each set of image axis scales.
%    imageaxistype  : type of imageaxisscale to use with each mode of data, values
%                     can be 'discrete' 'stick' 'continuous' 'none'.
%    imagemap       : reference for included pixels of an image. Calls
%                     function to display data.
%    imagedata      : reference for image data, holds no data but will call
%                     function to display folded data.
%    data           : actual data consisting of any Matlab array of class
%                     double, single, logical, or [u]int8/16/32.
%    label          : text labels for each row, column, etc of data.
%    labelname      : descriptive name for each set of labels.
%    axisscale      : axes scales for each row, column, etc of data.
%    axisscalename  : descriptive name for each set of axis scales.
%    axistype       : type of axisscale to use with each mode of data, values
%                     can be 'discrete' 'stick' 'continuous' 'none'.
%    title          : axis titles for each row, column, etc of data.
%    titlename      : descriptive name for each axis title.
%    class          : class indentifiers for each row, column, etc of data.
%    classname      : descriptive name for each set of class identifiers.
%    classlookup    : lookup table for text names for each numeric class.
%    classid        : a reference that assigns/returns a cell array of stings
%                     based on the .classlookup table.
%    include        : indices of rows, columns, etc to use from data (allows
%                     "exclusion" of data without hard-deletion)
%    userdata       : user defined content.
%    description    : text description of DataSet content.
%    history        : text description of modification history.
%    uniqueid       : a unique identifier given to this DataSet.
%    datasetversion : dataset object version.
%
% For more information on working with DataSet objects, see the methods: 
%    DATASET/SUBSREF and DATASET/SUBSASGN
% For more detail on DataSet functionality, see the DataObject documentation. 
%
%I/O: data = dataset(a);
%
%See also: DATASET/EXPLODE, DATASET/SUBSASGN, DATASET/SUBSREF

%Copyright Eigenvector Research, Inc. 2000-2019

%nbg 8/3/00, 8/16/00, 8/17/00, 8/30/00, 10/05/00, 10/09/00
%nbg added 5/11/01  b.includ = cell(nmodes,2); (this is different from
%    the previous version which used b.includ = cell(ndims) which didn't
%    follow the convention of different modes on different rows
%jms 5/30/01 added transposition of row-vector batch cell to column-vector
%nbg 10/07/01 changed version from 2.01 to 3.01
%jms 8/30/02 added validclasses string
%    added empty dataset construction
%jms 11/06/02 change version to 3.02
%    updated help
%jms 4/24/03 modified help (includ->include)
%    -renamed "includ" to "include"
%rsk 09/08/04 add image size and mode.

%Construct a dataset object template
b.name      = '';            %variable name   char
b.type      = '';            %data type       char
                               %   data  {default}
                               %   image
                               %   batch
b.author      = '';            %dataset author  char
b.date        = [];            %creation date
b.moddate     = [];            %last modified date
b.imagesize   = [];            %size of an image if type = image 
b.imagemode   = [];            %location of image spatial mode
b.data        = [];            %double array

%define valid classes for dataset of type "data" 
validclasses = {'double','single','logical','int8','int16','int32','uint8','uint16','uint32'};

if nargin==0
  nmodes    = 2;
elseif nargin>0
  if nargin>1;
    try
      a = cat(2,varargin{:});
    catch
      error('Cannot combine these items into a single DataSet object')
    end
  else
    a = varargin{1};
  end
  if isempty(a)
    nmodes  = 2;
    b.type  = 'data';        %default
  elseif any(strcmp(class(a),validclasses))
    nmodes  = ndims(a);
    b.type  = 'data';
    b.data  = a;
  elseif isa(a,'cell')
    if (size(a,1)>1)&(size(a,2)>1)
      error('Not set up for multidimensional cells.')
    else
      if size(a,2)>1; a=a'; end; %flip to be COLUMN vector
      nmodes  = ndims(a{1});     %number of modes for each cell
      csize   = size(a{1});
      csize   = csize(2:end);    %size of dimensions~=1
      if ~isnumeric(a{1})
        error('Batch DataSet objects can only be created from numeric types.');
      end
      if length(a)>1             %make certain that contents of all
        for ii=2:length(a)       % cells are same size except dim 1
          if ~isnumeric(a{ii})
            error('Batch DataSet objects can only be created from numeric types.');
          end
          csize2 = size(a{ii});
          csize2 = csize2(2:end);
          if any(csize2~=csize)
            error('All modes except 1 must be same size.')
          end
        end      
      end
      b.type  = 'batch';
      b.data  = a;
    end
  else
    error(['Unable to create a dataset to contain variables of class ' class(a)])
  end
end

b.label       = cell(2,2,1);   %empty cell
b.axisscale   = cell(2,2,1);   %empty cell
b.imageaxisscale   = cell(2,2,1);   %empty cell
b.title       = cell(2,2,1);   %empty cell
b.class       = cell(2,2,1);   %empty cell
b.include     = cell(2,2,1);   %empty cell

%empty cell, has same name as class so use second dimension for 'set' rather than 3rd.
b.classlookup = cell(2,1);
b.axistype    = cell(2,1);
b.imageaxistype    = cell(2,1);

for ii      = 1:nmodes
  b.label{ii,1,1}       = ''; b.label{ii,2,1}     = ''; %'Set 1';
  b.axisscale{ii,1,1}   = []; b.axisscale{ii,2,1} = ''; %'Set 1';
  b.imageaxisscale{ii,1,1}   = []; b.axisscale{ii,2,1} = ''; %'Set 1';
  b.title{ii,1,1}       = ''; b.title{ii,2,1}     = ''; %'Set 1';
  b.class{ii,1,1}       = []; b.class{ii,2,1}     = ''; %'Set 1';
  b.classlookup{ii,1}   = {}; %Assign empty to set 1, additional sets in second mode.
  b.axistype{ii,1}   = 'none'; %Assign empty to set 1, additional sets in second mode.
  b.imageaxistype{ii,1}   = 'none'; %Assign empty to set 1, additional sets in second mode.
end

if nargin==0
  b.include       = cell(2,2);  %empty cell with size = ndims x 1  %nbg changed 5/11/01
elseif ~isa(a,'cell')
  b.include       = cell(nmodes,2);  %nbg added 5/11/01
  for ii=1:nmodes
    b.include{ii} = [1:size(b.data,ii)]; 
  end
else
  b.include       = cell(nmodes,2);  %nbg added 5/11/01
  b.include{1}    = [1:length(b.data)];
  for ii=2:nmodes
    b.include{ii} = [1:size(b.data{1},ii)]; 
  end
  b.axisscale{1,1,1} = cell(length(a),1);
  b.imageaxisscale{1,1,1} = cell(length(a),1);
end
b.description    = '';       %character string
b.userdata       = [];       %userdata

b.datasetversion = '6.0';   %version number of the dataset object
                             %this is hidden (can't be set) but
                             %can be obtained using GET.
b.history      = cell(1,1);  %changed in SET
[tstamp,time] = timestamp;
b.history{1}   = ['Created by ' userinfotag ' ' tstamp];
b.uniqueid     = makeuniqueid(time);

if nargin==0
  data         = class(b,'dataset');
elseif isa(a,'dataset')
  data         = a;
elseif any(strcmp(class(a),validclasses)) | isa(a,'cell');
  data         = b; clear b
  data.name    = inputname(1);
  data.date    = clock;
  data.moddate = data.date;
  data         = class(data,'dataset');
else
  error(['Datasets can not be made from class ' class(a) ' variables'])
end

