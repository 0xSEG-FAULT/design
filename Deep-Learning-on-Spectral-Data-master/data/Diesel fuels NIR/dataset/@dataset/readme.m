%ReadMeDataSet
% This README lists the procedures for updating the DataSet
% object and includes a running list of the release dates,
% version (e.g. for DataSet h: h.datasetversion), and comments
% on changes from previous versions.
%
% Full documentation of the DataSet Object can be found at:
%
% http://www.wiki.eigenvector.com/index.php?title=DataSet_Object
%
%I) When updating the DATASET object:
%
%I.A) Version number (double) must be changed in:
% 1) constructor dataset.m
% 2) overloaded loadobj.m
%
%I.B) Version changes follow the convention:
%  for Version xx.yy.zz
% 1) xx is incremented if object fields are changed or added
%   a) update DISPLAY, HELP, SUBSREF, SUBSASGN, CAT
% 2) yy is incremented if object methods are changed or added (leading to
%    non-backwards compatible objects)
% 3) zz is incremented for bug-fixes and changes in methods which are
%    backwards compatible with previous released version.
%
%I.C) As of version 5.0, the following functions need to be modified if a
%  field is added to the DSO.
%    cat 
%    dataset
%    display
%    loadobj
%    permute
%    readmedataset
%    reshape
%    set
%    squeeze
%    subsasgn
%    subsref 
%
%II) Update notes:
%Release Dates  Version  Comments
%   12-04-12     6.0.2
%                        Add simple plot functionality
%                        Add search functionality
%                        Modify load object method to assure history field always reflects the load of an object
%                        Note in history when load object changes NaN classes to zeros
%                        Added updateset method to allow updating/adding label sets automatically without knowing which set index to use. 
%                        Added findset method to allow locating label sets automatically without knowing which set index to use. 
%                        Add error checking when assigning entire imageaxisscale field
%                        Allow assignment directly onto imageaxisscale
%                        Clarify that axistype is being shown in () after each axisscale
%                        Fix mistaken use of varargin. Threw error if trying to sortby a DSO when the DSO wasn't passed directly into sortby (e.g. through another function or after an subsref call)
%                        Discontinue throwing of error when user attempts to index into non-existent but otherwise valid set. Instead simply return an empty (of appropriate type). This makes set indexing more friendly (the set doesn't exist, but it CAN exist, so returning empty value is more valid than throwing error)
%                        Allow indexing by single index when ROW vector (e.g. x(1:5) when x is a row vector should return COLUMNS 1-5)
%                        More memory-friendly handling of in-place sub-assignment (portion) of data
%                        Direct subscripting assignments into dataset.data will operate without making a duplicate copy
%                        Improve history record of event including calling function
%                        Fix && to & in if statement. && can't handles more than a scalar.
%                        Fix of userinfotag (two locations:  @dataset/private and utilities) for problems when running with unusual directory names (e.g. parallels)
%                        Make sure indexes into DSOs are vectorized (to avoid accidental creation of matrices from things like classes)
%                        Add sparse, issparse, and full to DSO per helpdesk/ticket 942.
%                        Allow empty string to be interpreted as 'none' for axistype
%                        Change warning to give mode, set, and replaced value and REPEAT warning for each time we do this.
%                        Remove limitation on the warnings (always give it)
%                        loadobj.m
%                          -change logic to replace NaN's with zero ONLY if not currently used. otherwise use first unused negative value.
%                          -update class lookup table after changing classes
%                          -change evriwarndlg to be simple warning (and give each time a DSO is loaded with NaN's)
%                        subsasgn.m
%                          -minor change to wording of warning
%                          -change evriwarndlg to be simple warning
%                        Automatically convert NaN's in class assignments to zeros
%                        Add a permute (to put data in correct order if imagemode > 1 and useimagemode='off')
%Release Dates  Version  Comments
%   10-10-10     6.0.1
%                        Added ability to index into batch DSOs using cell notation (converts given batch to 'data' type) 
%                        Add string precedence for merging class lookup tables.
%                        Add virtual field for included data, mydata.data.include
%                        Fix for label/class direct indexing into dataset, returns empty if can't find.
%                        Fix for duplicate name values.
%                        Modified .include call to ONLY use indexing IF there is something excluded (more memory and time friendly)
%                        Fix for transposed include field.
%                        Fix for wrong new class number, was causing duplicate numeric values in class lookup table.
%                        Fix for removing last set.
%                        fix to allow correct indexing into imageaxisscale field with n-D images
%Release Dates  Version  Comments
%   09-08-09      6.0    
%                        Added "imageaxisscale", axis scale for images.
%                        Added "imageaxisscalename", axis scale name for images.
%                        Added "imageaxistype", axis scale type for images.
%                        Added "imagedataf" field to return "full" folded images. Where spacial modes are unfolded in the position of .imagemode. 
%                        Added "imagesizestr", returns image size as a string.
%                        Added "foldedsize", returns folded size of image.
%                        Added "foldedsizestr", returns folded size as a string.
%                        Added "size", returns dataset size as vector.
%                        Added "sizestr", returns dataset size as a string.
%                        Added "uniqueid" field, [read only] returns the unique identifier string for a given dataset.
%                        Added "anyexcluded" function, returns boolean true if any data has been excluded.
%                        Added "rmset" function, removes a specific set from a DSO field.
%                        Timestamps now added to history field entries.
%                        User login informaion now added to history field entries when datasets are read in from MAT or created new
%                        Ability to add comments to history field.
%                        Document .classlookup{}.assignstr functionality.
%                        Document .classlookup{}.assignval functionality.
%Release Dates  Version  Comments
%   01-23-09      5.2    
%                        Added new "anyexcluded" function.
%Release Dates  Version  Comments
%   11-18-08      5.1    
%                        Added new "reshape" function.
%                        Add sorting functions and editds GUI updates.
%                        Add "complex" notifier for when data ~isreal.
%                        Multiple fixes for class lookup table.
%                        Add timestamp to history field.
%                        Fix datatype displayed for class in disp.
%                        Add display for classid.
%                        Fix for DSO permute.
%                        Add classid handling to plsda and simca. 
%                        Dataset no longer uses PLS_Toolbox code.
%                        Allow passing of string array to assign classes (cell array is, however, preferred)
%                        Give caller info in history (if available).
%                        Don't add class 0 if classes are empty.
%                        Removing class from lookup table automatically reassigns given samples to class 0 (zero).
%                        Force sort of class lookup.
%                        Allow subscripting into DSO using labels or class names.
%                        Allow sub-indexed assignments on classid field.
%   3-02-07       5.0    Added overload support for:
%                           DISP  DOUBLE  ISEMPTY  NUMEL  
%                           SINGLE  SORTROWS  UNQIUE
%                         Having these overloaded allows operation with
%                           the Mathworks synonomous object on the path.
%                         NOTE: Mathworks Dataset object is NOT compatible
%                           with EVRI DataSet object.
%   1-10-07       5.0     Add class lookup table to allow string
%                           assignment/reference of classes. Information
%                           resides in .classlookup field and can be
%                           referenced as a cell array of stings with the
%                           .classid field. Functions in I.C-1 have been
%                           updated.
%                         Add axis type field to define axis type for given
%                           axis scale (dim,set). Information resides in
%                           .axistype field. Functions in I.C-1 have been
%                           updated.
%                         Add uminus function to support Unary minus. 
%   3-30-06       4.01    Updated manual, this document, and constructor/loader
%                         Added overload support for:
%                           LDIVIDE  RDIVIDE  TIMES  MINUS  PLUS
%                           REPMAT  SQUEEZE
%                         Added internal utility:  mergelabels.m
%                         Modified:
%                          CAT fixed bug associated with unequal non-empty
%                              "description" fields 
%                          DELSAMPS fixed behavior when all samples or variables deleted
%                                   fixed typo in help
%                          DISPLAYIMAGE fixed typo in help
%                          LOADOBJ fix update of old (pre 4.0) type=image
%                               data (imagesize values) 
%                          SUBSREF allow cell indexing into userdata field
%   3-09-05       4.0     updated manual, this document, and constructor/loader
%                         Enhanced Image mode support added: imagesize,
%                         imagemode, imagemap, imagedata fields.
%                         (Changes to dataset, display, loadobj, subsref,
%                         subsasgn, cat)
%                         CAT allow cat of empty DSOs (no effect)
%                           don't duplicate identical fields already in DSO
%                         LENGTH method added
%   8-31-04       3.04    updated manual, this document, and constructor/loader
%                         SUPSREF base vdim tests on ndims(x) code instead of
%                         ndims(x.data)
%                         SUBSASGN base vdim tests on ndims(x) code instead of
%                         ndims(x.data)
%                         SIZE fixed singleton index in last mode bug
%                         NDIMS revised to base on number of modes shown in
%                         include field
%                         DELSAMPS base vdim tests on ndims(x) code instead of
%                         ndims(x.data)
%                         CAT fixed single-slab to multi-slab bug
%                           fixed single-slab to multi-slab bug
%                           associated with titles
%                           fixed "empty fieldname stored as double
%                           instead of char" bug
%                           title name field better handled
%   7-28-03       3.03    updated manual, this document, and constructor/loader
%   7-24-03               DELSAMPS fixed bug associated with multiple title sets
%                         CAT fixed additional multi-way concat bugs
%   4-24-03               Renamed includ to include - still allow old use of "includ" 
%                         LOADOBJ fixed update from pre 1.05 datasets (field order problem) 
%   4-07-03               CAT fixed multi-way concat bugs
%                         PERMUTE, IPERMUTE, TRANSPOSE, CTRANSPOSE, NDIMS added as methods
%
%  11-08-02       3.02    updated manual, this document, and constructor/loader
%                         CAT fixed bugs when concatenating with different numbers of sets,
%                           allowed cat in dim > ndim (e.g. cat of slabs), history field note 
%                           of cat operation, allow cat of datasets and non-datasets, use name 
%                           and author from all datasets.
%   9-09-02               DELSAMPS make history notation for both soft AND hard deletions
%                           added "hard keep/reorder" option (flag==3)
%                         END method added to allow correct indexing into DataSet at top level
%                         SUBSASGN changed subscripting error message
%                         SUBSREF added object-level subscripting
%   8-30-02               DATASET added validclasses string, added empty dataset construction
%                         SIZE method added, gives size of .data field
%                         SUBSASGN removed extraneous tests of data class, revised error messages
%                           fixed various bugs associated with batch support
%                         SUBSREF allow extraction of all-modes and all-sets from fields (calls w/o vdim)
%                           revised error messages
%   5-21-02               LOADOBJ revised test logic to detect "same-fields but not same version #"
%                           added includ field orientation and test for includname presence
%                         SUBSASGN changed calls to sethistory and sethistory output format
%   4-16-02               SUBSASGN removed extraneous code in sethistory
%  11-06-01               CAT command added (generic concatenation), HORZCAT and VERTCAT now use CAT
%                         SUBSASGN allow empty assignment to axisscale and label fields

%Copyright Eigenvector Research, Inc. 2009-2019
%Licensee shall not re-compile, translate or convert "M-files" contained
% in PLS_Toolbox for use with any software other than MATLAB®, without
% written permission from Eigenvector Research, Inc.

%  10-19-01        3.01   updated the manual, changed INCLUD, SET, GET, and data classes
%                         added SUBSASGN and SUBSREF
%   6-06-01               VERTCAT fixed bug when combining two datasets with different # of includ sets
%   6-05-01               SUBSASGN fixed includ size bug (adding data to empty dataset created
%                         includ field with only 1 column)
%   5-31-01               GET revised as call to DATASET/SUBSREF
%                         SET revised as call to DATASET/SUBSASGN (centralized code)
%                         SUBSASGN added subscripting into fields for assignment
%                         fixed bug which kept batch data axisscales from being addressed correctly
%                         fixed bugs in data assign checking batch data size
%                         SUBSREF added subscripting into ALL fields
%   5-30-01               DELSAMPS added transposition of row-vector batch cell to column-vector
%   5-14-01               DELSAMPS added "hard delete" flag==2
%   5-11-01               DELSAMPS terminated line 67 x.includ{vdim} = setdiff(x.includ{vdim},inds) with ;
%                         added b.includ = cell(nmodes,2); (this is different from the
%                         previous version which used b.includ = cell(ndims) which didn't
%                         follow the convention of different modes on different rows
%                         DISPLAY improved include display, and fixed size bug for cells (lablel, etc.)
%                         VERTCAT adapted for use with ingle, uint8, etc.
%                         removed use of SET and GET
%                         changed to allow class sets in mode 1 to be augmented, empty classes set to zeros
%                         (original was: mode 1 class sets are made empty),
%                         loop data cancate over 2:length(i2) not 2:iv (removes bug)
%   5-09-01               DELSAMPS adapted for use with: single, uint8, etc.
%                         HORZCAT adapted for use with single, uint8, etc.
%                         removed use of SET and GET
%                         SUBSASGN added handling of other classes for data field
%                         SUBSREF adapted for use with single, uint8, etc.
%   5-03-01               SUBSREF added indexing into data using ()s (e.g. x.data(1:5,:) )
%   4-22-01               LOADOBJ updated version check #, fixed string comparison bug (version number as
%                         stored is STRING), fixed update code to copy non-dataset object first before updating,
%                         made version comparison & update incremental (do all which are needed),
%                         added set of DataSetVersion field
%   4-20-01               DELSAMPS fixed parentesis bug: changed "}" to ")"
%                         fixed index bug e.g. if includ = 6:10 and user asked to remove 8, it wouldn't
%                         remove anything BUT if user asked to remove 2, 7 would be removed)
%   4-12-01               SET fixed includ error if adding data to empty dataset
%                         SUBSASNG fixed includ error if adding data to empty dataset
%   4-11-01               SUBSASGN nearly the same as SET for datasets except added logic to take input like a
%                         SUBSASGN call and translate it into the correct info for the old SET routine
%                         fixed error in errors (missing [ ]s)
%                         modified error statements to make more sense with SUBSASGN call
%                         changed history logging function (renamed old one to keep it)
%                         SUBSREF converted from DATASET/GET
%                         added indexing into history vs. GET which doesn't allow that
%  10-10-00        2.01   DATASET add type = batch(cell), add includ {cell}, add history(cell)
%                         DISPLAY allow for type = batch(cell), add includ, history
%                         LOADOBJ, GET, SET add includ(cell) and history(cell)
%                         HORZCAT, VERTCAT not defined for batch
%                         added DELSAMPS method for soft delete (hard not yet available)
%  10-06-00        1.05   changed SET lines 218, 220, 222 to label, axisscale,
%                         axisscalename repspectively. changed help in GET.
%  09-01-00        1.04   changed SET to input a date when empty when using SET
%                         changed datasetdemo.m
%  08-30-00        1.03   beta,
%                         added moddate,
%                         added horizontal concatenation,
%                         vertical concatenation,
%                         changed display to not include 0x0 (must not
%                          be empty to see MxN)
%                         get moddate and datasetversion
%                         made moddate and date setable only if initially empty
%                         set moddate and datasetversion errors
%  08-17-00        0.02   beta,
%                         added SET field input can be cell,
%                         GET data can be data(:,1) (indexing capability)
%                         XPLDST overloaded for DATASET objects
%  08-16-00        0.01   beta, initial release
%  08-03-00        0.00   alpha
