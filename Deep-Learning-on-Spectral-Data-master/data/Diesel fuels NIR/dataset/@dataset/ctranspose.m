function out = ctranspose(in)
%DATASET/CTRANSPOSE Performs transpose on Dataset Object.
%
%I/O: xt = ctranspose(x)
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
[mytimestamp,out.moddate] = timestamp;
notes  = ['   % ' mytimestamp];
out.history(end) = {[z ' = ' z ''';' notes]};
