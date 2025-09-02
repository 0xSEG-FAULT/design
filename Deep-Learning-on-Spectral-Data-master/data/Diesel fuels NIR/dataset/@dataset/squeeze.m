function b = squeeze(a)
%DATASET/SQUEEZE  Remove singleton dimensions of a DataSet Object.
%   Returns a dataset object (b) with the same elements as (a) but
%   with all the singleton dimensions removed.  A singleton
%   is a dimension such that size(A,dim)==1.  2-D arrays are
%   unaffected by squeeze so that row vectors remain rows.
%I/O: B = squeeze(a)

% Copyright © Eigenvector Research, Inc. 2005-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.
%JMS

sz = size(a);

b = a;
if length(sz)>2;
  singleton = (sz==1);
  keep = ~singleton;
  forcetranspose = false;
  if sum(keep)==1
    %if we're going to end up dropping to only one dimension...
    toswitch = min(find(singleton));
    if toswitch<min(find(keep))
      forcetranspose = true;
    end
    singleton(toswitch) = false;
    keep = ~singleton;
  end
  resz = sz(keep);
  
  if ~strcmp(b.type,'image') | (strcmp(b.type,'image') & singleton(b.imagemode))
    %type= data or if type image and the imagemode was one of our singleton dims
    b.type = 'data';  %switch to type data
    b.imagesize = [];
    b.imagemode = [];
    b.imageaxisscale = cell(sum(keep),2,1);
    [b.imageaxisscale{:,2,:}] = deal('');
    [b.imageaxistype{keep,:}] = deal('none');
  end
  
  if length(resz)==1; resz(2) = 1; end
  
  %     data
  b.data = reshape(a.data,resz);
  %     label
  b.label = b.label(keep,:,:);
  %     axisscale
  b.axisscale = b.axisscale(keep,:,:);
  %     axistype
  b.axistype = b.axistype(keep,:);
  %     title
  b.title = b.title(keep,:,:);
  %     class
  b.class = b.class(keep,:,:);
  %     classlookup
  b.classlookup = b.classlookup(keep,:);
  %     include
  b.include = b.include(keep,:,:);
  
  %     history
  varname = inputname(1);
  if isempty(varname);
    varname = '?FunctionCall';
  end
  b.history{end+1} = ['squeeze(',varname,');'];
  
  if forcetranspose
    b = b';
  end
end
