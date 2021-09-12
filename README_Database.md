#How to use the Database object

The basic intuition is that each datatype must be called separately. For example, say we have an experiment that generates a light absorbance data, and we want to classify that instance of experiment into a cluster. In this case, light absorbance data (which may be in tuple form or the string name of the excel file elsewhere) is considered "Raw Data" and must be stored in that location only. The cluster, which may be represented as an integer, is considered "Dependent Variable 1". The Database object automatically detects the datatype of the first input and initializes the specific dataset accordingly. Once this is done, the datatype cannot be changed, so take caution.

The current version of Database supports the following datatype to be stored:

   * int, float
   * string
   * tuple
   * list, numpy array
   * images

Attempt to store other datatype will raise error. Refer to the helper function *datatype_extraction* in regards to the datatype limitations.

##Initialization
**DB = Database(dim_arr, filename, mode = 0, der = 0, dvars = 0, reset = False, sparse = True, ask = False, overwrite = True)**

###dim_arr

* specifies the dimensions formed by independent variables

###filename

* string name for the HDF5 file

###mode

* denotes the type for "Raw Data"
* if raw data is image, mode must be set to 1. In every other case, it must be set to 0 (currently supports int, float, tuple, list, array, and string)
* default is set to 0

###der

* specifies how many derived data can come out from the raw data
* for example, feature vector from images or the string name of the image file can be considered Derived Data 1 and Derived Data 2 if Raw Data is the image itself
* default is set to 0

###dvars

* specifies how many dependent variables are given
* in a simple case where Raw Data is dependent variable and there is only one dependent variable, treat Raw Data dataset as equal to dependent variable dataset. Otherwise, specify how many dependent variables we want
* default is set to 0 (signifying the simple case above)

##reset

* if set to true, it will erase all the previous data that the HDF5 might have been holding
* default is set to 0

##sparse

* if set to true, the temporary database is optimized to sparse-matrix; if set to false, it's optimized to dense-matrix
* defalut is set to True

##ask
* if set to true, whatever is given as filename variable is irrelevant (still needs input), and the terminal will prompt the user to input the name for the hdf5 file. If set to false, whatever's set as filename variable is the name of the hdf5 file.
* default is set to False

##overwrite
* if set to true, if data already exists for the given experiment, the data will be overwritten
* if set to false, the data will not be overwritten but rather saved in the next entry
* default is set to True for simplicity

###example uses

**Database([10, 10], "basic")**

   *Database with 2 independent variables (dimension of 10 * 10), name of "basic", and has the most simple 1 dependent variable which corresponds to raw data*

**Database([10, 8, 20], "foo", mode = 0, dvars = 4)**

   *Database with 3 independent variables (dimension of 10 * 8 * 20), name of "foo", has a simple Raw Data, and 4 dependent variables are derived from it*

**Database([5, 5], "foo2", mode = 1, der = 3, dvars = 2)**

   *Database with 2 independent variables (dimension of 5 * 5), name of "foo2", has image as raw data, has 3 kinds of derived data, and has 2 dependent variables*

**Database([10, 10], "foo3", ask = True)**

   *Database with 2 independent variables (dimension of 10 * 10), the terminal will prompt the user to input the name of the database which will be the file name ("foo3" is irrelevant)

##Data Storage

**DB.store(arr, destination = "Raw Data")**

Note that the first call of store function to a specific destination will initialize the dataset with the appropriate type. For instance, calling store to destination = "Derived Data 3" with data of type float list will initialize dataset inside HDF5 with type float list, and this is immutable.

###arr

* denotes the array of tuples which has data and index

###destination

* denotes where the data will be stored
* default is set to "Raw Data" dataset
* if user wants to store to first dependent variable, set this to "Dependent Variable 1; if second derived data, "Derived Data 2", etc.

###example uses

**DB.store([(34, 0, 2), (12, 6, 4)])**

   *store 34 at location (0, 2) and 12 at location (6, 4) in "Raw Data"*

**DB.store([(-11, 8, 7, 3)], destination = "Derived Data 12")**

   *store -11 at location (8, 7, 3) of 12th Derived Data set*

##Data Storge (to HDF5)

While Database object has a capability to automatically store data to HDF5 if specific condition is satisfied (such as if the temporary space gets too large), user can explicitly store data to HDF5 with the following function call.

**DB.flush()**

No input needed; this call will take all data inside temporary space and write it to the HDF5 for permanent storage.

##Data Retrieval
**DB.retrieve(getfrom, location = "Raw Data", archived = False, flag = "one", entry = -1)**

###getfrom

* denotes the location of the data contrived of ivar specification; the form depends on the flag which is addressed below

###location

* denotes the location of the data
* default is set to "Raw Data"; analogous concept of data storage

###archived

* if False, retrieve from tempDB. Else, retrieve from HDF5
* defualt is set to False

###flag

* different "versions" of retrieval
* "one" retrieves just one element; the form should be in tuple of independent variables like (0, 0, 3)
* "arr" retrieves from a list of ivar tuples; will return not only the data but also the tuple related to the data; getfrom should be list of ivar tuples
* "all" retrieves all data we have; getfrom can be anything but has to be fed in
* "ivar" retrieves all data we have for a specific element of a specific independent variable; for 4th element of first independent variable, getfrom should be (0, 3) as it's zero-indexed
* "entry" retrieves all data of the specified experiment as a tuple (Raw Data, Derived Data, and Dependent Variable in that order)

###entry

* in non-overwrite mode, the "data" that's returned is an array of all entries. If a user wants to extract a specific entry, they can manipulate this variable
* must only be manipulated in non-overwrite mode
* type int; this isn't 0-indexed so if the user wants to extract the data from the first entry, this variable must be set to 1; this is consistent with how data is stored as HDF5 groups
* if entry goes beyond what's been recorded, retrieve function will return np.nan in that position 
* defualt is set to -1 which will just return arrays as data

example uses

**DB.retrieve((8, 2))**
   *retrieve data at location (8, 2) in "Raw Data"*

**DB.retrieve((3, 5), location = "Dependent Variable 4")**
   *retrieve data at location (3, 5) in 4th Dependent Variable set*

**DB.retrieve("foo", flag = "all")**
   *retrieve all data we have*

**DB.retrieve("hi", flag = "all", archived = True)**
   *retrieve all data we have but from HDF5*

**DB.retrieve((1, 7), flag = "ivar")**
   *retrieve all data we have in 8th element of the second independent variable*

**DB.retrieve([(0, 0, 0), (5, 3, 2)], flag = "arr")**
   *retrieve data in (0, 0, 0) and (5, 3, 2) from "Raw Data"*

**DB.retrieve((2, 5), flag = "entry", entry = 2)**
   *retrieve data of second entry of (2, 5) experiment as a tuple*

##Image Viewing script

Trying to retrieve image using DB.retrieve will give image object. To visually see the image, run HDF5_imageviewer.py and follow instructions to see images stored in Raw Data dataset.

##Campaign Object Storage

**DB.campaignflush(campaignObject)**

Recommended to be called within BioActive; this function will take all historical attributes of campaignObject and store it within the HDF5 along with all the data inside the temporary space.

**DB.flagcheck(campaignObject)**

Recommended to be called within BioActive in the beginning; this function will check if the database already exists, and if so, let the user to decide if they want to continue form previous progress. If the user decideds to do so, all the stored data and historical attributes of the campaign object is loaded back in, and asks the user to input the data from the previous protocol. In any other cases, it will start the campaign from the beginning.

##Batch Storage

**DB.storehistory(arr)**

###arr

* array of indices (in number form) that represents one batch
* each successive call to storehistory will store the batches at different "round"
* assumed that the batch size is fixed

###example uses

**DB.storehistory([3, 7, 5, 4])**

   *store batch size of 4 with indices listed above*
   *converting the numbers into tuple of indicies is handled internally*

##Batch Retrieval

**DB.gethistory(batch)**

###batch

* integer that represents the round
* 0-indexed, so the "first round" is equivalent to DB.gethistory(0)

###example uses

**DB.gethistory(12)**

   *retrieves the list of indices that the active learner requested at round 13*

##Limitations of current version

* when you re-run the campaign, the batch storage will start storing the new indicies after the previous round and make it keep growing, so delete the previous database; I'll fix this issue in near future
* the batch index storing resizes for every 100 batches, so say if 250 batch indices are stored, then the next 50 will all be filled with (0, 0, ..)
* current flush condition is set to False, so nothing is stored into HDF5 until flush() is explicitly called