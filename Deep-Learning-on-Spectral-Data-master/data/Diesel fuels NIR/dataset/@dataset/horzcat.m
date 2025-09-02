function y = horzcat(varargin);
%DATASET/HORZCAT Horizontal concatenation of DataSet objects.
%  [a b] is the horizontal concatenation of DataSet objects
%  (a) and (b). Any number of dataset objects can be
%  concatenated within the brackets. For this operation
%  to be defined the following must be true:
%    1) All inputs must be valid dataset objects.
%    2) The DataSet 'type' fields must all be the same.
%    3) Concatenation is along the second dimension. The first
%       and, for multiway, remaining dimension sizes must match.
%
%  This is similar to matrix concatenation, but each field
%  is treated differently. In structure notation this is:
%    z.name      = a.name;
%    z.type      = a.type;
%    z.author    = a.author
%    z.date      date of concatenation
%    z.moddate   date of concatenation
%    z.data      = [a.data b.data c.data ...];
%    z.label     mode 2 label sets are concatenated, and new
%                label sets are created for all other modes
%    z.axisscale mode 2 axisscale sets are made empty, and new
%                axisscale sets are created for all other modes
%    z.title     new label sets are created
%    z.class     mode 2 class sets are made empty, and new
%                class sets are created for all other modes
%    z.description concatenates all descriptions
%    z.userdata  if more than one input has userdata it is
%                filled into a cell array, if only one input
%                has userdata it is returned, else it is empty
%
%I/O: z = [a, b, c, ... ];
%  
%See also: DATASET, DATASET/CAT, DATASET/VERTCAT

%Copyright Eigenvector Research, Inc. 2000-2019
%jms 11/06/01 -converted to use cat

y = cat(2,varargin{:});

return

