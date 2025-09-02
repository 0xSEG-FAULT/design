function img = displayimage(imgin, isize, mode, useimagemode)
%DISPLAYIMAGE folds unfolded image data based on original size.
%     Given a matrix of image data or a DataSet Object (DSO) with image
%     data in the .data field, this function will attempt to fold the
%     data into its original image.
%
%     INPUTS:
%           imgin = matrix of image data or a DSO containing image data in
%                    the .data field.
%            size = [1xN] vector of image size.
%            mode = [scalar] mode where image data resides.
%            useimagemode = ['on'|{'off'}] put folded image at .imagemode
%                           location. When 'off' spatial dims are first.
%     OUTPUTS:
%             img = DSO with onfolded image data in the .data field.
%
%Example:
%
%I/O: img = displayimage(imgin);
%I/O: img = displayimage(imgin,useimagemode);
%I/O: img = displayimage(imgin, size, mode);
%I/O: img = displayimage(imgin, size, mode,useimagemode);
%I/O: displayimage demo
%
%See also:

%Copyright Eigenvector Research, Inc. 2004-2019
%Licensee shall not re-compile, translate or convert "M-files" contained
% in PLS_Toolbox for use with any software other than MATLAB®, without
% written permission from Eigenvector Research, Inc.

if nargin == 1
  if isa(imgin,'dataset')
    idata = imgin.data;
    isize = imgin.imagesize;
    mode = imgin.imagemode;
    useimagemode = 'off';
  else
    error('DISPLAYIMAGE can only take an image DSO or image data with size and mode.');
  end
elseif nargin == 2
  if isa(imgin,'dataset') && ischar(isize)
    useimagemode = isize;
    idata = imgin.data;
    isize = imgin.imagesize;
    mode = imgin.imagemode;
  else 
    error('DISPLAYIMAGE can only take an image DSO or image data with size and mode.');
  end
elseif nargin == 3
  idata = imgin;
  useimagemode = 'off';
end

insize = size(idata);
indims = 1:ndims(idata);
modloc = find(ismember(indims,mode));
if strcmp(useimagemode,'on')
  reshapev = [insize(1:modloc-1) isize insize(modloc+1:end)];
else
  % must permute the idata array to put imagemode data first
  imagemodeFirst = [modloc [1:modloc-1] [(modloc+1):length(insize)]];
  idata = permute(idata, imagemodeFirst);
  reshapev = [isize insize(1:modloc-1) insize(modloc+1:end)];
end
img = reshape(idata,reshapev);
