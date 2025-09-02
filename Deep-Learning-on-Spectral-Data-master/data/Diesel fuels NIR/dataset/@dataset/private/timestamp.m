function [tstamp,time] = timestamp(time)
%TIMESTAMP Returns detailed timestamp string for DataSet object history field
% Outputs a string-formatted timestamp for a DSO's history field. Also
% returns the timestamp used in "clock" vector format.
%
%I/O: [tstamp,time] = timestamp;        %returns current timestamp
%I/O: [tstamp,time] = timestamp(time);  %returns timestamp for given clock vector

%Copyright Eigenvector Research, Inc. 2009-2019
%Licensee shall not re-compile, translate or convert "M-files" contained
% in PLS_Toolbox for use with any software other than MATLAB®, without
% written permission from Eigenvector Research, Inc.

persistent lasttime stampparts

if nargin==1
  %if user passed in time vector, do NOT cache it, just create timestamp
  tstamp = [ datestr(time,'dd-mmm-yyyy') ' ' datestr(time,'HH:MM') sprintf(':%06.3f',time(end))];
  return
end

time = clock;

if isempty(lasttime) | any(lasttime(1:3)~=time(1:3))
  stampparts{1} = datestr(time,'dd-mmm-yyyy');
end
if isempty(lasttime) | any(lasttime(4:5)~=time(4:5))
  stampparts{2} = datestr(time,'HH:MM');
end
    
tstamp = [ stampparts{1} ' ' stampparts{2} sprintf(':%06.3f',time(end))];
lasttime = time;
