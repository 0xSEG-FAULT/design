function explode(sdat,txt)
%DATASET/EXPLODE Extracts variables from a DataSet object.
%  EXPLODE writes the fields of a DataSet input
%  (sdat) to variables in the workspace with the same
%  variable names as the field names.
%  The optional string input (txt) appends a string
%    (txt) to the variable outputs.
%
%I/O: explode(sdat,txt)
%
%Example: explode(h,'01')
%
%See also: DATASET/GET, DATASET/SUBSREF

%Copyright Eigenvector Research, Inc. 2000-2019
%NBG 8/17/00, 9/1/00
%JMS 4/24/03 -simplified code, -rename "includ"

if nargin<2,     txt = []; end

fields = fieldnames(sdat);

for thisfield=fields';
  if ~ismember(thisfield,{'datasetversion'});    %as long as it isn't a feild we don't want to explode

    dat = getfield(sdat,thisfield{:});
    
    if strcmp(thisfield{:},'includ');
      thisfield = {'include'}; %rename 'includ'
    end
    assignin('caller',[thisfield{:},txt],dat)    %store it
    
  end
end

%removed JMS 4/25/03
% ii     = strmatch('datasetversion',fields);
% ij     = 1:size(fields,1);
% ij     = ij(find(ij~=ii));
% 
% for ii=ij
%   eval(['dat = sdat.',fields{ii,:},';'])
%   if isempty(txt)
%     assignin('caller',fields{ii,:},dat)
%   else
%     assignin('caller',[fields{ii,:},txt],dat)
%   end
% end
