function display(data)
%DATASET/DISPLAY Command window display of a dataobj object.

%Copyright Eigenvector Research, Inc. 2000-2019
%nbg 8/3/00, 8/16/00, 8/17/00, 8/30/00, 10/10/00
%nbg 5/11/01 improved include display, and fixed size bug for cells (lablel, etc.)
%jms 10/19/01 cleaned up display, added "mode x" labels, added full-display of .title info
%jms 4/24/03 -renamed "includ" to "include"
%jms 3/1/07 -moved main code into dataset/disp.m

%display variable name
disp(' ');
disp([inputname(1),' = '])

disp(data)

disp(['      OTHER: <a href="matlab:disp(''>> classsummary(' inputname(1) ')'');classsummary(' inputname(1) ')">[View Class Summary]</a>']);
