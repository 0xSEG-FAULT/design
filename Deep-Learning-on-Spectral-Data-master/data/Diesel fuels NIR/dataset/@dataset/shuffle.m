function [ varargout ] = shuffle(dsoIn,vDim,inVect)
%SHUFFLE randomizes the order of the dataset in a specific mode.
%   Shuffles the rows/columns of the DSO and returns a shuffled DSO, a
%   vector of the shuffled indexes, and a vector of indexes which can be
%   used for undoing the shuffle.
%
%   INPUTs:
%           dsoIn = dataset object to be shuffled.
%           vDim  = (int) specified dimension to be shuffled.
%
%   OPTIONAL INPUT:
%           inVect = (int vector) inVect is the sorting order used to
%                    shuffle dsoIn.
%                    There are two special uses for inVect: repeat a
%                    shuffle, or undo a shuffle. To repeat a shuffle on
%                    another dso, inVect should be the rndVect of a
%                    previous shuffle. To undo a shuffle, dsoIn should be
%                    a previously shuffled dataset and inVect its
%                    respective undoVect.
%
%   OUTPUT:
%           dsoOut   = the shuffled dataset object.
%           rndVect  = a vector of randomized indexes of dsoOut
%           undoVect = a vector containing a sorted indexes that can be
%           used to undo the shuffling of dsoOut.
%
%I/O: dsoOut = shuffle(dsoIn,vDim);
%I/O: [dsoOut,randV,undoV] = shuffle(dsoIn,vDim);
%I/O: shuffledLikeLastDsoOut = shuffle(newDso,vDim,lastRandV);
%I/O: UnshuffledDso = shuffle(LastDsoOut,vDim,lastUndoV);
%
%See also: DATASET/SORTBY DATASET/SORTCOLS DATASET/SORTROWS

%Copyright Eigenvector Research, Inc. 2017-2019

% Get input argument
if nargin<2
  error('Two input arguments required (dataset and mode to shuffle).')
end

if nargin<3
  inVect = [];
end

inVect = inVect(:)';

% Check vDim
if ~(isnumeric(vDim))
  error('The specified mode is not valid.');
end
if (vDim<1) | (vDim>ndims(dsoIn))
  error('The specified dimension must be greater than 0 but not exceed the number of modes in the dataset.');
end

% Check lastRandVect
if ~(isempty(inVect))
  if(length(inVect) ~= size(dsoIn,vDim))
    error('The lengths of the input vector and dataset must agree on the specified dimension.');
  end
  if ~((isnumeric(inVect)) | (any(inVect)<1) | (all(inVect) < length(inVect)))
    error('Input vector must be valid indexes of dataset.');
  end
end

if ~isempty(inVect)
  % Sort a dso using inVect (shuffle a 'newDso' (if inVect is a randVect),or unshuffle a 'shuffledDso' (if inVect is a undoVect)
  dsoOut = delsamps(dsoIn,inVect,vDim,3);
  rndVect=[];
  undoVect=[];
else
  % Generate randomization vector.
  idxs = size(dsoIn,vDim);
  rnd = rand(1,idxs);
  [rndVals,rndVect] = sort(rnd,'ascend');
  
  % Generate/update undo vector.
  [undoVals,undoVect] = sort(rndVect,'ascend');
  % if ~(isempty(lastRandVect))
  %   [undoVals,undoVect] = sort(lastRandVect(rndVect),'ascend');
  % else
  %   [undoVals,undoVect] = sort(rndVect,'ascend');
  % end
  
  
  clear rnd_vals undoVals;
  
  % Shuffle DSO
  dsoOut = delsamps(dsoIn,rndVect,vDim,3);
  
end

% Update history
thisName = inputname(1);
if isempty(thisName)
  thisName = ['"' dsoIn.name '"'];
end
if isempty(thisName)
  thisName = 'unknown_dataset';
end
if (isempty(inVect))
  vectName = '[]';
else
  vectName = inputname(3);
  if isempty(vectName)
    vectName = ['[1 x ' num2str(length(inVect)) ']'];
  end
end
caller = '';
try
  [ST,I] = dbstack;
  if (length(ST)>1)
    [a,b,c] = fileparts(ST(end).name);
    caller = [' [' b c ']'];
  end
catch
end
cmd = ['x = shuffle(' thisName ', ' num2str(vDim) ',' vectName ')'];
dsoOut.history = [dsoOut.history(1:end-1) { [cmd '  % ' timestamp caller]}];

% Set output arguments (1-3, dsoOut, shuffleVect, undoVect)
varargout{1} = dsoOut;
varargout{2} = rndVect;
varargout{3} = undoVect';


function test

load wine

% shuffle mode 1
dsoOut = shuffle(wine,1);
[wine.data(:,1),dsoOut.data(:,1)]
% shuffle mode 2
dsoOut = shuffle(wine,2);
[wine.data(1,:),dsoOut.data(1,:)]
% shuffle mode 3 (error)
dsoOut = shuffle(wine,3);

% shuffle mode 1, get randV & undoV
[dsoOut,randV,undoV] = shuffle(wine,1);
disp('undo random with undoV');
[wine.data(:,1) dsoOut.data(undoV,1)]
disp('redo random with randV')
[wine.data(randV,1) dsoOut.data(:,1)]

% undo shuffle with shuffle() & undoV
[dsoUndo,~,~] = shuffle(dsoOut,1,undoV);
disp('undo random with shuffle and undoV');
[wine.data(:,1) dsoUndo.data(:,1)]

% redo shuffle with shuffle() & randV on new dataset
[dsoRand,~,~] = shuffle(wine,1,randV);
disp('Redo random with shuffle and randV');
[dsoRand.data(:,1) dsoOut.data(:,1)]
