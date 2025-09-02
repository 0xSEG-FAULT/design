function disp(data)
%DATASET/DISP Display the dataset contents without printing the array name.
% Overload of the standard DISP command.
%I/O: disp(data)

%Copyright Eigenvector Research, Inc. 2007-2019
%jms 3/1/07 Created from dataset/display.m

%display field names/values (class/sizes)
%data.name, .type, .author, .date, .moddate
disp(['       name: ',data.name])
disp(['       type: ',data.type])
disp(['     author: ',data.author])
s         = data.date;
if isempty(s)
  disp(['       date:'])
else
  disp(['       date: ',datestr(datenum(s(1),s(2),s(3),s(4),s(5),s(6)),0)])
end
s         = data.moddate;
if isempty(s)
  disp(['    moddate:'])
else
  disp(['    moddate: ',datestr(datenum(s(1),s(2),s(3),s(4),s(5),s(6)),0)])
end

%data.data
s         = (['       data: ',int2str(size(data.data,1))]);
for ii=2:ndims(data.data)
  s       = [s,'x',int2str(size(data.data,ii))];
end
realstr = '';
if ~isreal(data.data)
  realstr = ' (complex)';
end
disp([s,' [',class(data.data), realstr, ']'])

%data.imagesize/imagemode if type = image.
if strcmp(data.type, 'image')
  %Consturct sting of sizes.
  sizestr = '';
  for i = 1:length(data.imagesize);
    sizestr = [sizestr num2str(data.imagesize(i)) 'x'];
  end
  sizestr = sizestr(1:end-1); %Remove last x.

  disp(['  imagesize: ',sizestr])
  disp(['  imagemode: ',num2str(data.imagemode)])
end

%data.label
[m1,n,n2] = size(data.label);
s         = [int2str(size(data.label,1)),'x',int2str(size(data.label,3))];
disp(['      label: {',s,'} [array (char)]'])
%individual label sizes        
if strcmp(class(data.label),'cell')
  s2      = ' ';
  s2      = s2(ones(m1,15));
  for ii=1:n2     %cell slabs
    s     = '';
    for ij=1:m1   %cell rows
      if ii==1; prefix = ['Mode ' num2str(ij) '  ']; else; prefix = ''; end
      if isempty(data.label{ij,1,ii})
        s1  = [prefix '[',data.label{ij,2,ii},': ] '];
      else
        s1  = [prefix '[', data.label{ij,2,ii},': ',int2str(size(data.label{ij,1,ii},1)), ...
            'x',int2str(size(data.label{ij,1,ii},2)),'] '];
      end
      s   = strvcat(s,s1);
    end
    s2    = [s2 s];
  end
end
disp(s2)

%data.axisscale
[m1,n,n2] = size(data.axisscale);
s         = [int2str(size(data.axisscale,1)),'x',int2str(size(data.axisscale,3))];
disp(['  axisscale: {',s,'} [vector (real)] (axistype)'])
%individual data.scale sizes        
if strcmp(class(data.axisscale),'cell')
  s2        = ' ';
  s2        = s2(ones(m1,15));
  for ii=1:n2     %cell slabs
    s       = '';
    for ij=1:m1   %cell rows
      if ii==1; 
        prefix = ['Mode ' num2str(ij) '  ']; 
      else
        prefix = ''; 
      end
      if isempty(data.axistype{ij,ii})
        suffix = '';
      else
        suffix = ['(' data.axistype{ij,ii} ')'];
      end
      if isempty(data.axisscale{ij,1,ii})
        s1    = [prefix '[',data.axisscale{ij,2,ii},': ] ' suffix ' '];
      else
        s1    = [prefix '[',data.axisscale{ij,2,ii},': ',int2str(size(data.axisscale{ij,1,ii},1)), ...
            'x',int2str(size(data.axisscale{ij,1,ii},2)),'] ' suffix ' '];
      end
      s     = strvcat(s,s1);
    end
    s2      = [s2 s];
  end
end
disp(s2)

%data.title
[m1,n,n2]   = size(data.title);
s           = [int2str(size(data.title,1)),'x',int2str(size(data.title,3))];
disp(['      title: {',s,'} [vector (char)]'])
%individual data.title sizes        
if strcmp(class(data.title),'cell')
  s2        = ' ';
  s2        = s2(ones(m1,15));
  for ii=1:n2     %cell slabs
    s       = '';
    for ij=1:m1   %cell rows
      if ii==1; prefix = ['Mode ' num2str(ij) '  ']; else; prefix = ''; end
      if isempty(data.title{ij,1,ii})
        s1    = [prefix '[',data.title{ij,2,ii},': ] '];
      else
        if size(data.title{ij,1,ii},2) > 20;
          %too many characters to actually display
          s1    = [prefix '[',data.title{ij,2,ii},': ',int2str(size(data.title{ij,1,ii},1)), ...
              'x',int2str(size(data.title{ij,1,ii},2)),'] '];
        else
          s1    = [prefix '[',data.title{ij,2,ii},': ''',data.title{ij,1,ii},'''] '];
        end
      end
      s     = strvcat(s,s1);
    end
    s2      = [s2 s];
  end
end
disp(s2)

%data.class
[m1,n,n2]   = size(data.class);
s           = [int2str(size(data.class,1)),'x',int2str(size(data.class,3))];
disp(['      class: {',s,'} [vector (double)]'])
%individual data.class sizes        
if strcmp(class(data.class),'cell')
  s2        = ' ';
  s2        = s2(ones(m1,15));
  for ii=1:n2     %cell slabs
    s       = '';
    for ij=1:m1   %cell rows
      if ii==1; prefix = ['Mode ' num2str(ij) '  ']; else; prefix = ''; end
      if isempty(data.class{ij,1,ii})
        s1    = [prefix '[',data.class{ij,2,ii},': ] '];
      else
        s1    = [prefix '[',data.class{ij,2,ii},': ',int2str(size(data.class{ij,1,ii},1)), ...
            'x',int2str(size(data.class{ij,1,ii},2)),'] '];
      end
      s     = strvcat(s,s1);
    end
    s2      = [s2 s];
  end
end
disp(s2)



%data.classlookup
[m1,n]   = size(data.classlookup);
s        = [int2str(size(data.classlookup,1)),'x',int2str(size(data.classlookup,2))];
disp(['    classid: {',s,'} [cell of strings]'])

%disp(['classlookup: {',s,'} [cell array (nx2 integer,char)]'])
% %individual data.classlookup sizes        
% if strcmp(class(data.classlookup),'cell')
%   s2        = ' ';
%   s2        = s2(ones(m1,15));
%   for ii=1:n     %cell slabs
%     s       = '';
%     for ij=1:m1   %cell rows
%       if ii==1; prefix = ['Mode ' num2str(ij) '  ']; else; prefix = ''; end
%       if isempty(data.classlookup{ij,ii})
%         s1    = [prefix '[',data.classlookup{ij,ii},': ] '];
%       else
%         s1    = [prefix '[',data.class{ij,2,ii},': ',int2str(size(data.classlookup{ij,ii},1)), ...
%                 'x',int2str(size(data.classlookup{ij,ii},2)),'] '];
%       end
%       s     = strvcat(s,s1);
%     end
%     s2      = [s2 s];
%   end
% end
% disp(s2)

%data.include (nbg added 5/11/01)
[m1,n,n2]   = size(data.include);
s           = [int2str(m1),'x',int2str(n2)];
disp(['    include: {',s,'} [vector (integer)]'])
%individual data.class sizes        
if strcmp(class(data.include),'cell')
  s2        = ' ';
  s2        = s2(ones(m1,15));
  for ii=1:n2     %cell slabs
    s       = '';
    for ij=1:m1   %cell rows
      if ii==1; prefix = ['Mode ' num2str(ij) '  ']; else; prefix = ''; end
      if isempty(data.include{ij,1,ii})
        s1    = [prefix '[',data.include{ij,2,ii},': ] '];
      else
        s1    = [prefix '[',data.include{ij,2,ii},': ',int2str(size(data.include{ij,1,ii},1)), ...
            'x',int2str(size(data.include{ij,1,ii},2)),'] '];
      end
      s     = strvcat(s,s1);
    end
    s2      = [s2 s];
  end
end
disp(s2)

%description
if isempty(data.description)
  disp(['description:'])
else
  disp(['description: ',data.description(1,:)])
  if size(data.description,1)>1
    for ii=2:size(data.description)
      disp(['             ',data.description(ii,:)])
    end
  end
end

%history
if isempty(data.history)
  disp(['    history:'])
else
  s          = int2str(size(data.history,1));
  disp(['    history: {',s,'x1 cell} [array (char)]'])
end

%data.userdata
%note, include some info about contents
s = '   userdata: ';
if isempty(data.userdata)
  disp(s)
else
  s = [s,int2str(size(data.userdata,1))];
  for ii=2:ndims(data.userdata)
    s = [s,'x',int2str(size(data.userdata,ii))];
  end
  disp([s,' [',class(data.userdata),']'])
end
