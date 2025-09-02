function out = mergelabels(dso_a,dso_b,operator);
%DATASET/MERGELABELS combine all label fields of two same-size datasets
% Combines labels, axisscales, classes, etc in all modes of two identically
% sized DataSet objects. Mostly used as a supporting function for simple
% math calculations. Output is first DataSet object with all the unique
% labels, axisscales, classes, and titles contained in both input dataset
% objects. 
%
% The optional input (operator) will combine labels from the two DataSets
% concatenated with the operator string. If omitted, the non-common labels
% in the two DataSets will be stored in separate label sets in the output
% DataSet. For example, a string of '+' will combine all textual labels
% with a plus sign between them: 
%      'mylabel_a'   and   'mylabel_b'  yeilds  'mylabel_a+mylabel_b'
% instead of storing 'mylabel_a' and 'mylabel_b' in two separate label
% sets.
%
%I/O: DSO_A = mergelabels(DSO_A,DSO_B)
%I/O: DSO_A = mergelabels(DSO_A,DSO_B,operator)

% Copyright © Eigenvector Research, Inc. 2006-2019
% Licensee shall not re-compile, translate or convert "M-files" contained
%  in PLS_Toolbox for use with any software other than MATLAB®, without
%  written permission from Eigenvector Research, Inc.
%JMS

if ndims(dso_a)~= ndims(dso_b) | any(size(dso_a)~=size(dso_b))
  out = dso_a;
  return
end

%use concatenate code to get combined labels, axisscales, classes, etc.
modes = 1:ndims(dso_a);
beencopied = [];
out = dso_a;
for mode = modes;
  try
    temp = cat(mode,dso_a,dso_b);
    for k = setdiff(modes,mode);
      if ~ismember(k,beencopied)  %don't copy mode if already done (this happens only in n-way datasets)
        out  = copydsfields(temp,out,k); %copy fields in all other modes
        beencopied(end+1) = k;           %note we've already done this mode
      end
    end
  catch
    %probably couldn't do this because it is a scalar combining with a
    %non-scalar, just ignore this.
    %OR!! we are running without copydsfields (a PLS_Toolbox function)
  end
end

%see if we can combine labels with specially supplied operator string
% (this operates on LABELS ONLY)
if nargin>2
  for mode = modes;
    try
      %combine labels specially with supplied operator string
      for setind=1:size(out.label,3);
        for k = 1:2; %do for both labels (k=1) and labelnames (k=2)
          %combine labels with operator (if not empty)
          if size(dso_a.label,3)>=setind;
            lbl_a = dso_a.label{mode,k,setind};
          else
            lbl_a = [];
          end
          if size(dso_b.label,3)>=setind;
            lbl_b = dso_b.label{mode,k,setind};
          else
            lbl_b = [];
          end
          if isempty(lbl_a) | isempty(lbl_b)
            out.label{mode,k,setind} = '';
            continue; %skip this set
          end
          out.label{mode,k,setind} = [lbl_a repmat(operator,size(lbl_a,1),1) lbl_b];
        end
      end
    catch
      %probably couldn't do this because it is a scalar combining with a
      %non-scalar, just ignore this.
    end
  end
end

