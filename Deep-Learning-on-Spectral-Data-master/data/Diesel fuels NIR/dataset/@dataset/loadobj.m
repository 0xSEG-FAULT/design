function data = loadobj(x)
%DATASET/LOADOBJ Load dataset object and update to present version.
%  If the loaded dataset object is newer than present dataset
%  object constructor it is converted to a structure.

%Copyright Eigenvector Research, Inc. 2000-2019
%nbg 8/16/00, 8/17/00, 8/30/00
%jms 4/22/01
%  -updated version check #
%  -fixed string comparison bug (version number as stored is STRING)
%  -fixed update code to copy non-dataset object first before updating
%  -made version comparison & update incremental (do all which are needed)
%  -added set of DataSetVersion field
%jms 5/21/02
%  -revised test logic to detect "same-fields but not same version #"
%  -added includ field orientation and test for includname presence
%jms 11/06/02
%  -updated dataset version number
%jms 4/8/03
%  -history field now added as CELL to old (pre 1.05) datasets
%  -reorder fields to match current dataset field order
%jms 4/24/03
%  -renamed "includ" to "include", revised dataset object version
%jms 9/29/05
%  -fix update of old (pre 4.0) type=image data (imagesize values)
%rsk 01/09/07
%  -update to version 5, add lookup table for class and axis type.

%to change version number change "pv = " and "data.datasetversion ="
junk  = dataset;
pv    = str2double(junk.datasetversion);         %present version of constructor
if isa(x.datasetversion,'char');
  xv    = str2num(x.datasetversion);  %version of the loaded dataset object
else
  xv    = x.datasetversion;           %version of the loaded dataset object
end;

if xv>pv
  
  if exist('struct2dataset')
    data = struct2dataset(struct(x));
  else
    disp('Warning: Loaded dataset object newer than present constructor.')
    disp('  Dataset object converted to structure.')
    data = struct(x);
  end
  
elseif xv<=pv
  
  data = x;     %copy structure into what will become our dataset
  
  %update fields
  if xv<=0.02 %update from 0.02
    data.moddate = x.date;
    data.datasetversion = '1.05';
  end;
  
  if xv<=1.05 %update from 1.05
    data.includ  = cell(ndims(x.data),2);
    for ii=1:length(data.includ)
      data.includ{ii} = [1:size(x.data,ii)];
    end
    data.history = {'Loaded from dataset object version <= 1.05.'};  %made cell 4/8/03
    data.datasetversion = '1.06';  %changed to 1.06 05/21/02
  end;

  %check for non-cell history field
  if ~iscell(data.history)
    data.history = {};
  end
  
  if xv<=3.00  %update to 3.01
    %test for wrong includ field orientation and missing labels
    if size(data.includ,1) == 1 & size(data.includ,2) == ndims(x.data);
      data.includ = data.includ';
    end
    if size(data.includ,2) == 1;
      data.includ{ndims(x.data),2} = [];    %add empty labels
    end
    data.datasetversion = '3.01';  %changed to 3.01 05/21/02
    data.history = [data.history {'Loaded from dataset object version <= 3.01.'}];   %added 4/8/03
  end
  
  if xv<3.02
    data.datasetversion = '3.02';  %update to 3.02 is easy
  end
  
  if xv<3.03   %jms added 4/24/03
    data.include = data.includ;
    data = rmfield(data,'includ');
    data.datasetversion = '3.03';  %update to 3.03
    data.history = [data.history {'Loaded from dataset object version <= 3.03.'}];   %added 4/24/03
  end
  
  if xv<3.04 %rsk added 8/31/04
    data.datasetversion = '3.04';
  end
  
  if xv<4.0 %rsk added 9/09/04
    data.imagesize = [];            %size of an image if type = image
    data.imagemode = [];            %location of image spatial mode
    if strcmp(data.type, 'image');
      %Fill in defaults if type image.
      data.imagesize = [size(data.data,1) 1];
      data.imagemode = 1;
    end
    data.datasetversion = '4.0';
    data.history = [data.history {'Loaded from dataset object version <= 4.0.'}];   %added 9/09/04
  end
  
  if xv<4.01 %rsk added 9/09/04
    data.datasetversion = '4.01';
  end
  
  if xv<5.0 %rsk added 11/02/06
    %Map out axistype to size of axisscale.
    asize = size(x.axisscale);
    d1 = asize(1);
    if length(asize)==3
      d2 = asize(3);
    else
      d2 = 1;
    end
    data.axistype  = repmat({'none'},[d1 d2]);
    
    %Add class lookup table.
    for ii = 1:ndims(data.data)
      for jj = 1:size(data.class,3)
        %Loop over sets.
        if ~isempty(data.class{ii,1,jj})
          data.classlookup{ii,jj} = classtable(data.class{ii,1,jj});
        else
          data.classlookup{ii,jj} = {};
        end
      end
    end
    data.history = [data.history {'Loaded from dataset object version <= 5.0.'}];
    data.datasetversion = '5.0';
  end
  
  if xv<5.1
    %Update to 5.1
    %make sure empty classlookup fields are cells
    clu = data.classlookup;
    for ii = 1:size(clu,1);
      for jj = 1:size(clu,2);
        if isempty(clu{ii,jj}) & ~iscell(clu{ii,jj})
          data.classlookup{ii,jj} = {};
        end
      end
    end
    data.history = [data.history {'Loaded from dataset object version <= 5.1.'}];
    data.datasetversion = '5.1';
  end
  
  if xv<5.2
    %Update to 5.2
    data.history = [data.history {'Loaded from dataset object version <= 5.2.'}];
    data.datasetversion = '5.2';
  end
  
  if xv<6.0
    %Update to 6.0
    data.history = [data.history {'Loaded from dataset object version <= 6.0.'}];

    data.uniqueid = '';
    
    %Add image axis scale/type
    asize = size(x.axisscale);
    
    %If type = image, then build according to image size.
    if strcmp(data.type,'image')
      isize = length(x.imagesize);
      data.imageaxisscale = cell(isize,2,1);
      data.imageaxistype  = repmat({'none'},[isize 1]);
    else
      %Just add empties same size as normal axisscale.
      d1 = asize(1);
      if length(asize)==3
        d2 = asize(3);
      else
        d2 = 1;
      end
      data.imageaxisscale = cell(asize(1),2,1);
      data.imageaxistype  = repmat({'none'},[d1 d2]);
    end
    data.history = [data.history {'Added image axis scale.'}];
    data.datasetversion = '6.0';
    
  end
  
  if xv<=6.0
    %for any version 6.0 or earlier DSOs do this (NOTE: we assume that for
    %later versions, these tests do NOT need to be done)
    % Convert any NaN class values to zero values
    warning_given = false;
    for idim = 1:size(data.class,1)
      for iset = 1:size(data.class,3)
        val = data.class{idim, 1, iset};
        if ~isempty(val) & isnumeric(val)
          i = isnan(val);
          if any(i)
            %determine what value we should replace with
            rep = 0;
            while any(val(~i)==rep);
              %if it isn't in use...
              rep = rep-1;
            end
            %replace them...
            data.class{idim,1,iset}(i) = rep;
            %and manage lookup table
            lu = data.classlookup{idim,iset};
            lu = lu(~isnan([lu{:,1}]),:);
            if rep~=0 | ~ismember(0,[lu{:,1}])
              lu = [{rep} {'Class NaN'};lu];
            end
            data.classlookup{idim,iset} = lu;
            data.history = [data.history {sprintf('Load Object: Removed NaNs from class mode %i set %i',idim,iset)}];
            %give warning
            warning('EVRI:DSOClassNaN','DataSet class cannot contain "NaN". Replacing NaNs (mode %i, set %i) with "%i".',idim,iset,rep);
          end
        end
      end
    end
  end
  
  if isempty(data.uniqueid)
    %assure uniqueid is ALWAYS populated (do this test for ALL versions of
    %the DSO since it could happen if the user creates a MAT file manually)
        %generate a unique ID for this old DSO
    name = '';
    for catfield = {'name' 'author'};
      str = data.(catfield{:});
      str(~ismember(str,['@' 'A':'Z' 'a':'z' '0':'9'])) = '_';
      while any(findstr(str,'__'));
        str = strrep(str,'__','_');
      end
      str = str(1:min([end 40]));
      while ~isempty(str) & str(end)=='_'
        str(end) = [];
      end
      if isempty(str)
        str = 'unnamed';
      end
      name = [name '@' str];
    end
    name = name(2:end); %drop leading @
    uid = [name '@' datestr(data.moddate,30) '.' sprintf('%0.3i',round(mod(data.moddate(end),1)*100))];
    data.uniqueid = uid;
    data.history = [data.history {['Assigned UniqueID: ' uid]}];
  end

  if ~isa(data,'dataset');
    %make sure fields are in correct order (4/8/03)
    new = struct(dataset);
    for k=fieldnames(data)';
      new = setfield(new,k{:},getfield(data,k{:}));
    end
    %and call it a dataset
    data = class(new,'dataset');
  end
  
end

%Always add users "loaded" stamp to history field
data.history = [data.history {['=== Loaded By ' userinfotag ' ' timestamp]}];
  
