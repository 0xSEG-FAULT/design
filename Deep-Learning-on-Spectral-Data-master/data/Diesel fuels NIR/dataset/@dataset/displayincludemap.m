function imgincludemap = displayincludemap(imgin)
%DISPLAYINCLUDEMAP creates a logical map of included image data.
% Given a DataSet Object of type = 'image', this function will build a
% logical map of included pixels
%
% INPUTS:
%   imgin = a DSO of type = 'image'. 
% OUTPUTS:
%   imgincludemap = logical matrix of included pixels.
%
%I/O: imgincludemap = displayincludemap(imgin);
%
%See also: DATASET, DISPLAYIMAGE

%Copyright Eigenvector Research, Inc. 2004-2019
%Licensee shall not re-compile, translate or convert "M-files" contained
% in PLS_Toolbox for use with any software other than MATLAB®, without
% written permission from Eigenvector Research, Inc.

if nargin == 1
  if ~isa(imgin,'dataset')
    error('DISPLAYINCLUDEMAP can only take an image DSO.');
  end
else
  error('DISPLAYINCLUDEMAP can only take an image DSO.');
end

imgincludemap = false(imgin.imagesize);
imgincludemap(imgin.include{imgin.imagemode}) = 1;
