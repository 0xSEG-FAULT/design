function x = repmat(varargin)
%DATASET/REPMAT Replicate and tile a DataSet object.
% Overload of REPMAT function - see repmat for full I/O and help. 
% B = repmat(A,M,N) creates a large DataSet B consisting of an M-by-N
%      tiling of copies of A. The size of B is [size(A,1)*M, size(A,2)*N].
% B = REPMAT(A,[M N]) accomplishes the same result as repmat(A,M,N).
% B = REPMAT(A,[M N P ...]) tiles the DataSet A to produce a
%      multidimensional DataSet B composed of copies of A. The size of B is
%      [size(A,1)*M, size(A,2)*N, size(A,3)*P, ...].
%
%I/O: B = repmat(A,M,N) 
%I/O: B = REPMAT(A,[M N]) 
%I/O: B = REPMAT(A,[M N P ...]) 

% Copyright © Eigenvector Research, Inc. 2006-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.
%JMS


if nargin==1;
  error('Not enough input arguments');
end

if nargin==2;
  sz = varargin{2};
else
  sz = cat(2,varargin{2:end});
end

x = varargin{1};
for mode = 1:length(sz);
  temp = x;
  for nreps = 2:sz(mode);
    x = cat(mode,x,temp);
  end
end
