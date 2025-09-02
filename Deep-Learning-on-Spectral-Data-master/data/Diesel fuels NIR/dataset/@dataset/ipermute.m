function out = ipermute(in,order)
%DATASET/IPERMUTE Inverse permute array dimensions of a Dataset Object.
%  Rearranges the dimensions of (b) so that executing permute(a,order)
%  will give (b) again. The dataset produced has the same values of A but
%  the order of the subscripts needed to access any particular element are
%  rearranged as specified by ORDER. The elements of ORDER must be a
%  rearrangement of the numbers from 1 to N.
%  All informational fields are also reordered as necessary.
%  
%  permute and IPERMUTE are a generalization of transpose (.') for N-D
%  arrays.
%
%I/O: a = ipermute(b,order)
%
%See also: PERMUTE

%Copyright Eigenvector Research, Inc. 2003-2019

if any(order<1) | length(order)<max(order)
  error('ORDER contains an invalid permutation index.');
end

temp(order) = 1:max(order);
out  = permute(in,temp);

%put entry into history field
z = inputname(1);
if isempty(z); z = 'ans'; end
out.history(end) = {[z ' = ipermute(' z ',[' num2str(order) ']);']};
