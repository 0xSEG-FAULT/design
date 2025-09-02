function out = userinfotag
%USERINFOTAG Returns user and computer-specific string tag for history fields
%
%I/O: userinfo = userinfotag

%Copyright Eigenvector Research, Inc. 2009-2019
%Licensee shall not re-compile, translate or convert "M-files" contained
% in PLS_Toolbox for use with any software other than MATLAB®, without
% written permission from Eigenvector Research, Inc.

persistent info

if isempty(info);
  info = '';
  wrn = warning;
  warning('off');
  try
    if ispc
      info = sprintf('%s@%s', getenv('USERNAME'), getenv('COMPUTERNAME'));
    else
      [status, uname] = system ('whoami');
      [status, cname] = system ('hostname -s');
      info = sprintf('%s@%s', uname(uname~=10), cname(cname~=10));
    end
  catch
    info = '';
  end
  if isempty(info)
    info = 'Unknown_user@Unknown_computer';
  end
  warning(wrn);  
  info(info==10) = [];
end

out = info;
