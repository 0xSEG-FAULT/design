function id = makeuniqueid(time)
%MAKEUNIQUEID Generates a unique ID to use on a DataSet.
% This function returns a string which is almost certainly unique. It
% comprises the user's login name, the computer's name, the timestamp (in
% miliseconds) when the ID was generated and an additional random number.
%
% The only conditions under which this will not be unique is if the system
% clock is at the exact same moment in time (in miliseconds) and the random
% number generator happens to be in the exact same state. As this is nearly
% impossible, the uniqueness of this string is virtually guaranteed.
%
%I/O: id = makeuniqueid
%I/O: id = makeuniqueid(time)

%Copyright Eigenvector Research, Inc. 2009-2019
%Licensee shall not re-compile, translate or convert "M-files" contained
% in PLS_Toolbox for use with any software other than MATLAB®, without
% written permission from Eigenvector Research, Inc.

if nargin<1
  time = clock;
end
msec = sprintf(':%06.3f',time(end)); msec = msec(4:end);
rnd  = char('0' + floor(rand(1,5)*10));
id   = [userinfotag '@' datestr(time,30) msec rnd];

