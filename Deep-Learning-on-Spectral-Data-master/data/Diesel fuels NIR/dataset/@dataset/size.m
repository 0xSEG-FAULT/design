function varargout = size(x,dim)
%DATASET/SIZE Size of DataSet object.
%  Returns size of data stored in a DataSet object. See built-in Matlab
%  function SIZE for more information
%  Inputs are the dataset of interest (x), and the optional
%  dimension for which size information is desired (dim). 
%  With a single output (D), a vector of sizes of each dimension of x.data
%   is returned or the length of the single dimension specified by (dim).
%   With multiple outputs, each output is the value of the consecutive
%   dimensions.
%I/O: D = size(x,dim)
%I/O: [M,N] = size(x)

%Copyright Eigenvector Research, Inc. 2002-2019
%JMS 8/30/02
%jms 11/7/03 -fixed singleton index in last mode bug

if nargout == 0;
  myout = 1;
else
  myout = nargout;
end

if nargin>1;
  [varargout{1:myout}] = size(x.data,dim);
else
  if myout == 1;
    %special case - gets around problems with singleton index in last mode
    %  (e.g. when sz should = [3 4 1] we get [3 4] from size(x.data) )
    sz = ones(1,ndims(x));
    temp = size(x.data);
    sz(1:length(temp)) = temp;
    varargout = {sz};
  else
    [varargout{1:myout}] = size(x.data);
  end
end  
  
