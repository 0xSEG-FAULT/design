function out = transpose(in)
%DATASET/TRANSPOSE Performs transpose on Dataset Object.
%
%I/O: xt = transpose(x)
%
%See also: PERMUTE

%Copyright Eigenvector Research, Inc. 2003-2019

if length(size(in))>2
  error('Transpose on ND array is not defined.')
end

out = permute(in,[2 1]);

%put entry into history field
z = inputname(1);
if isempty(z); z = 'ans'; end
out.history(end) = {[z ' = ' z '.'';']};
