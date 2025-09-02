echo on
%DATASETDEMO Demonstrates use of the dataset object.
%  This demonstration illustrates the creation and
%  manipulation of dataset objects. Functions that
%  are demonstrated include: DATASET, GET, SET,
%  ISA, and EXPLODE. 
echo off
 
%Copyright Eigenvector Research, Inc. 2000-2019
%nbg 8/17/00, 8/31/00
%jms 7/20/2003
 
echo on
 
% You may wish to save and clear your workspace before
% running this demo. (Press control-c to abort now)
 
pause
%-------------------------------------------------
% First, a quick explanation as to why a standard dataset
% object is necessary at all. The idea stems from
% object oriented programming and is a way to collect
% all relevant information for a data set into a single
% object rather than in multiple variables. It is a way
% to maintain the integrity of a data SET not just a
% data array. The SET includes the data array, a name,
% labels for the samples and variables, numerical axis
% scales (e.g. wavelength or time stamp), etc. Using
% the standard also streamlines function input/output.
%
% The dataset object includes a 'userdata' field to
% allow extensibility. For example, a user might include
% a cell array in the 'userdata' to allow for virtually
% unlimited additional "fields" to be included in the
% dataset object.
 
pause
%-------------------------------------------------
% UPDATES
% Eigenvector Research, Inc. will provide free updates that
% can be downloaded from our site at www.eigenvector.com.
% Note that the object can be modified by users. However,
% we encourage requests for modifications and enhancements
% to be sent to Eigenvector Research who will maintain
% and distribute an official version of the dataset object.
 
pause
%-------------------------------------------------
% The function DATASET can be used to create a dataset object.
% But first data will be constructed in the workspace.
 
t    = [0:0.1:10]';
x    = [cos(t) sin(t) exp(-t)];
vars = {'cos(t)';'sin(t)';'exp(-t)'};
 
whos
 
pause
%-------------------------------------------------
% This data set includes three variables
% t          101x1  double vector of time scales, and
% vars       3x1    cell array of variable/column labels.
% x          101x3  double array of data.
 
% A data object can be constructed in 2 ways. The simplest
% is shown first.
 
pause
%-------------------------------------------------
h   = dataset(x)
 
pause
%-------------------------------------------------
% Note that the object has several fields for labels, axis scales
% dimension/mode titles, etc. Additionally the field 'name' has 
% the name of the original data array, the field 'date' contains
% the creation date, the field 'data' contains the data, and the
% field 'moddate' contains the creation date. All other fields are
% filled with empty arrays of the appropriate class.
 
% The function ISA returns a 'True' for the following
 
isa(h,'dataset')
 
pause
%-------------------------------------------------
% Alterntatively an empty dataset object g can be created by
    g = dataset;
% and the data fields set by
 
    g.data = x;
    g.name = 'x';
    g
 
pause
%-------------------------------------------------
% As discussed below, the data field needs to be filled before
% the label, axisscale, title, and class.
 
pause
%-------------------------------------------------
% Labels are placed in the appropriate field by using SET.
% The I/O format is: x.field{vdim}=value
 
h.author   = 'Data Manager';        %sets the author field
h.label{2} = vars;                  %sets the labels for columns = dimension 2
h.labelname{2} = 'Variables';       %sets the name of the label for columns
h.axisscale{1} = t;                 %sets the axis scale for rows = dimension 1
h.axisscalename{1} = 'Time';        %sets the name of the axis scale for rows
h.title{1}         = 'Time (s)';    %sets the title for rows
h.titlename{1}     = 'Time Axis';   %sets the titlename for rows
h.title{2} = 'f(t)';                %sets the title for columns
h.titlename{2} = 'Functions';       %sets the titlename for columns
 
% The input {vdim} corresponds to the dimension/mode for the labels.
% Both character and cell arrays can be input. Also note that the 
% title for the rows includes the units e.g. this could be used
% for a plot label.
 
% The size of the input label and axisscale values must be
% consistent with the input data size and they can not be
% set unless the data field is assigned first. The data
% field must be assigned before label, labelname, axisscale,
% axisscalename, title, titlename, class, or classname can
% be set.
 
% Note that the dimension is also known as the mode (row, columns,
% tubes, etc.). Here the word "dimension" follows the MATLAB
% convention outlined in the NDIMS function. However, we will
% try to use "mode" to avoid misinterpretations between the
% mode, dimensionality of each mode, or the pseudo-rank of the data.
 
pause
%-------------------------------------------------
% The contents of the dataset object can be viewed by typing
% the dataset variable name at the command line.
 
h
 
pause
%-------------------------------------------------
% Note that the fields 'name', 'type', 'author', 'date',
% 'moddate','data', 'label', 'axisscale', and 'title' now 
% have field values while fields 'class', 'description',
% and 'userdata' are empty.
 
% Also, note that in this example there are no labels for 
% the first mode, and no axisscale for the second mode.
 
pause
%-------------------------------------------------
% Set names such as 'labelname' don't have to be used but
% can be used to help distinguish the contents when multiple
% label sets are used. 
% Multiple sets can be created by adding a set number to the
% assignment statement immediately after the vdim number.
% For example, to input a second set of labels use:
 
h.label{2,2}     = {'COS(t)';'SIN(t)';'EXP(-t)'};
h.labelname{2,2} = 'Variables2';
h
 
pause
%-------------------------------------------------
% Values of a specific dataset object fields can be retrieved by
% simply providing the name of the DataSet, the name of the field,
% and, in {}s, any mode and set information (if necessary). For example:
 
name   = h.name
 
author = h.author
 
varlbl = h.label{2} 
 
pause
%-------------------------------------------------
% Inspection of the new variables in the workspace (name, author,
% and varlbl) show that they have been assigned the same values
% as the respective fields of the dataset object.
 
pause
%-------------------------------------------------
% This can also be used to plot variables:
 
figure
plot(h.axisscale{1},h.data)
xlabel(h.title{1})
legend(h.label{2},'Location','Best')
 
pause
%-------------------------------------------------
% or just plot a single variable
 
figure
plot(h.axisscale{1},h.data(:,2))
xlabel(h.title{1})
ylabel(h.title{2})
 
pause
%-------------------------------------------------
% Parts of the original DataSet can be extracted into a new DataSet by
% indexing into the DataSet object itself. For example, the data, labels,
% axisscales, etc for just the second variable can be extracted
% into a new DataSet using:
 
hsub = h(:,2);   %h and hsub are both datasets
 
pause
%-------------------------------------------------
% Looking at hsub, note the size of the data and label fields
 
hsub
 
pause
%-------------------------------------------------
% It may be of interest to extract all of the dataset object
% field data into the workspace in one step.
% The function EXPLODE can be used to extract all of the fields
% from a dataset object and place them in the workspace. E.g.
 
pause
%-------------------------------------------------
explode(h,'_A')
 
% places all the fields into the workspace and appends the
% text '_A' onto the the variable names. The optional text
% appended to the variable name allows multiple dataset objects
% to be extracted and differentiated.
 
whos
 
pause
%-------------------------------------------------
% For more information see help on
% DATASET, DATASET/SET, DATASET/GET, and DATASET/EXPLODE.
%
% Completion of DATASETDEMO.
 
echo off
