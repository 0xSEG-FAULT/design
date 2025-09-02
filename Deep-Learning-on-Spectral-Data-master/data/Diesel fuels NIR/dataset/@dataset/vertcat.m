function y = vertcat(varargin);
%DATASET/VERTCAT Vertical concatenation of DataSet objects.
%  [a; b] is the vertical concatenation of DataSet objects
%  (a) and (b). Any number of DataSet objects can be
%  concatenated within the brackets. For this operation
%  to be defined the following must be true:
%    1) All inputs must be valid dataset objects.
%    2) The Dataset 'type' fields must all be the same.
%    3) Concatenation is along the first dimension. The second
%       and, for multiway, remaining dimension sizes must match.
%
%  This is similar to matrix concatenation, but each field
%  is treated differently. In structure notation this is:
%    z.name      = a.name;
%    z.type      = a.type;
%    z.author    = a.author
%    z.date      date of concatenation
%    z.moddate   date of concatenation
%    z.data      = [a.data; b.data; c.data; ...];
%    z.label     mode 1 label sets are concatenated, and new
%                label sets are created for all other modes
%    z.axisscale mode 1 axisscale sets are made empty, and new
%                axisscale sets are created for all other modes
%    z.title     new label sets are created
%    z.class     mode 1 class sets are concatenated with
%                empty classes set to zeros, and new
%                class sets are created for all other modes
%    z.description concatenates all descriptions
%    z.includ    mode 1 inclue sets are concatenated with
%                empty classes set to zeros, and new
%                class sets are created for all other modes
%    z.userdata  if more than one input has userdata it is
%                filled into a cell array, if only one input
%                has userdata it is returned, else it is empty
%
%I/O: z = [a; b; c; ... ];
%
%See also: DATASET, DATASET/CAT, DATASET/HORZCAT

%Copyright Eigenvector Research, Inc. 2000-2019
%jms 11/06/01 -converted to use cat

y = cat(1,varargin{:});

return

