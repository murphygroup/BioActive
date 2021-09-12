import numpy as np
import h5py
from PIL import Image
import io
import sys
from scipy import sparse
from os import path

# Author: Seung Yun "Simon" Lee
# Date: June 5, 2019
# defines Database object that works both as a temporary storage space and permanent storage space in accordance with
# BioActvive specification
# note: refer to README_Database.md for example uses and explanation of variables

# TempDB and TempDB both serves as storage option in temporary space, and is
# automatically initialized when Database object is initiated. 

# TempDB class is optimized for dense-matrix 
class TempDB:
    def __init__(self, dim_arr, der, dvars, avars, overwrite):
        self.dim_arr = dim_arr
        self.der = der
        self.dvars = dvars
        self.avars = avars
        self.overwrite = overwrite

        if overwrite:
            self.tempdata = np.full(dim_arr, np.nan, dtype = object)
            self.tempder = []
            self.tempdvars = []
            self.tempavars = []
            if der != 0:
                for i in range(der):
                    self.tempder.append(np.full(dim_arr, np.nan, dtype = object))
            if dvars != 0:
                for i in range(dvars):
                    self.tempdvars.append(np.full(dim_arr, np.nan, dtype = object))
            if avars != 0:
                for i in range(avars):
                    self.tempavars.append(np.full(dim_arr, np.nan, dtype = object))
        else:
            self.tempdata = np.empty(dim_arr, dtype = object)
            for ind in zip(*np.where(self.tempdata == None)):
                self.tempdata[ind] = []
            self.tempder = []
            self.tempdvars = []
            if der != 0:
                for i in range(der):
                    temp = np.empty(dim_arr, dtype = object)
                    for ind in zip(*np.where(temp == None)):
                        temp[ind] = []
                    self.tempder.append(temp)
            if dvars != 0:
                for i in range(dvars):
                    temp = np.empty(dim_arr, dtype = object)
                    for ind in zip(*np.where(temp == None)):
                        temp[ind] = []
                    self.tempdvars.append(temp)
            if avars != 0:
                for i in range(avars):
                    temp = np.empty(dim_arr, dtype = object)
                    for ind in zip(*np.where(temp == None)):
                        temp[ind] = []
                    self.tempavars.append(temp)

        self.tempind = Index_storage(dim_arr, der, dvars, avars, overwrite)
        self.resetind = Index_storage(dim_arr, der, dvars, avars, overwrite)

    def write(self, tup, destination):
        data = tup[0]
        ind = tup[1:]

        if isinstance(data, list):
            data = np.asarray(data)

        if self.overwrite:
            if destination == "Raw Data":
                self.tempdata[ind] = data
            elif destination[:12] == "Derived Data":
                self.tempder[int(destination[13:]) - 1][ind] = data
            elif destination[:18] == "Dependent Variable":
                self.tempdvars[int(destination[19:]) - 1][ind] = data
            elif destination[:19] == "Associated Variable":
                self.tempavars[int(destination[20:]) - 1][ind] = data
        else:
            if destination == "Raw Data":
                self.tempdata[ind].append(data)
            elif destination[:12] == "Derived Data":
                self.tempder[int(destination[13:]) - 1][ind].append(data)
            elif destination[:18] == "Dependent Variable":
                self.tempdvars[int(destination[19:]) - 1][ind].append(data)
            elif destination[:19] == "Associated Variable":
                self.tempavars[int(destination[20:]) - 1][ind].append(data)

        self.tempind.store_ind(ind, destination)
        self.resetind.store_ind(ind, destination) 

    def show(self, ind, location = "Raw Data"):
        if location == "Raw Data":
                data = self.tempdata
        elif location[:12] == "Derived Data":
                data = self.tempder[int(location[13:]) - 1]
        elif location[:18] == "Dependent Variable":
                data = self.tempdvars[int(location[19:]) - 1]
        elif location[:19] == "Associated Variable":
            data = self.tempavars[int(location[20:]) - 1][ind]

        for idx in ind:
                data = data[idx]

        return data

    def exists(self, ind, location):
        return self.tempind.hasdata(ind, location)

    def reset(self):
        self.resetind = Index_storage(self.dim_arr, self.der,
                self.dvars, self.avars, self.ovewrite)


# TempDB_Sparse class is optimized for sparse-matrix
class TempDB_Sparse:
    def __init__(self, dim_arr, der, dvars, avars, overwrite, continuous=False):
        self.dim_arr = dim_arr
        self.der = der
        self.dvars = dvars
        self.avars = avars
        self.overwrite = overwrite
        self.continuous = continuous

        self.tempdata = {}
        self.tempder = {}
        self.tempdvars = {}
        self.tempavars = {}
        if der != 0:
            self.tempder = []
            for i in range(der):
                self.tempder.append(dict())
        if dvars != 0:
            self.tempdvars = []
            for i in range(dvars):
                self.tempdvars.append(dict())
        if avars != 0:
            self.tempavars = []
            for i in range(avars):
                self.tempavars.append(dict())

        if not continuous:
            self.tempind = Index_storage(dim_arr, der, dvars, avars, overwrite)
            self.resetind = Index_storage(dim_arr, der, dvars, avars, overwrite)
        else:
            self.tempind = Value_storage(dim_arr, der, dvars, avars, overwrite)
            self.resetind = Index_storage(dim_arr, der, dvars, avars, overwrite)

    def write(self, tup, destination):
        data = tup[0]
        ind = tup[1:]

        if isinstance(data, list):
            data = np.asarray(data)

        if self.overwrite:
            if destination == "Raw Data":
                self.tempdata[ind] = data
            elif destination[:12] == "Derived Data":
                self.tempder[int(destination[13:]) - 1][ind] = data
            elif destination[:18] == "Dependent Variable":
                self.tempdvars[int(destination[19:]) - 1][ind] = data
            elif destination[:19] == "Associated Variable":
                self.tempavars[int(destination[20:]) - 1][ind] = data
        else:
            if destination == "Raw Data":
                if ind not in self.tempdata:
                    self.tempdata[ind] = []
                self.tempdata[ind].append(data)
            elif destination[:12] == "Derived Data":
                if ind not in self.tempder[int(destination[13:]) - 1]:
                    self.tempder[int(destination[13:]) - 1][ind] = []
                self.tempder[int(destination[13:]) - 1][ind].append(data)
            elif destination[:18] == "Dependent Variable":
                if ind not in self.tempdvars[int(destination[19:]) - 1]:
                    self.tempdvars[int(destination[19:]) - 1][ind] = []
                self.tempdvars[int(destination[19:]) - 1][ind].append(data)
            elif destination[:19] == "Associated Variable":
                if ind not in self.tempavars[int(destination[20:]) - 1]:
                    self.tempavars[int(destination[20:]) - 1][ind] = []
                self.tempdvars[int(destination[20:]) - 1][ind].append(data)

        self.tempind.store_ind(ind, destination)
        self.resetind.store_ind(ind, destination)

    def show(self, ind, location = "Raw Data"):
        if isinstance(ind, (np.ndarray, list)):
            ind = tuple(ind)
        try:
            if location == "Raw Data":
                data = self.tempdata[ind]
            elif location[:12] == "Derived Data":
                data = self.tempder[int(location[13:]) - 1][ind]
            elif location[:18] == "Dependent Variable":
                data = self.tempdvars[int(location[19:]) - 1][ind]
            elif location[:19] == "Associated Variable":
                data = self.tempavars[int(location[20:]) - 1][ind]
        except KeyError:
            data = np.nan

        return data

    def exists(self, ind, location):
        return self.tempind.hasdata(ind, location)

    def reset(self):
        self.resetind = Index_storage(self.dim_arr, self.der, self.dvars, self.avars, self.overwrite)

class Value_storage:
    def __init__(self, dim_arr, der, dvars, avars, overwrite):
        self.dim_arr = dim_arr
        self.vars = len(dim_arr)
        self.overwrite = overwrite
        if overwrite:
            self.storeind = [{} for i in range(1+der+dvars+avars)] #storeind will be a dict, key = ivars, val = boolean
        else:
            self.storeind = [{} for i in range(1+der+dvars+avars)] #storeind will be a dict, key = ivars, val = number of entries
        self.locations = ["Raw Data"]
        for i in range(der):
            self.locations.append("Derived Data " + str(i + 1))
        for i in range(dvars):
            self.locations.append("Dependent Variable " + str(i + 1))
        for i in range(avars):
            self.locations.append("Associated Variable " + str(i + 1))
        
    # transforms a numpy indices into integer index of listTuples
    def transform_ind(self, ind):
        total = 0
        for i in range(self.vars):
            total += ind[i] * np.prod(self.dim_arr[i+1:])
        return int(total)

    # transforms an integer index of listTuples into indices of numpy array
    def transform_num(self, num):
        base = []
        for i in range(self.vars):
            ind = int(num / np.prod(self.dim_arr[i+1:]))
            num -= ind * np.prod(self.dim_arr[i+1:])
            base.append(ind)
        return tuple(base)

    def transform_location(self, location):
        return self.locations.index(location)

    def store_ind(self, ind, location):
        locationIndex = self.transform_location(location)
        if self.overwrite:
            self.storeind[locationIndex][ind] = True
        else:
            self.storeind[locationIndex][ind]  = self.storeind.get(ind, 0)+1

    def store_inds(self, inds, location = "Raw Data"):
        for ind in inds:
            self.store_ind(ind, location)

    # returns number of elements that's been stored
    # see if better optimization is possible
    def numelems(self, location = "Raw Data"):
        return sum(list(self.storeind.values()))

    # returns true if a data exists in that location
    def hasdata(self, ind, location):
        locationIndex = self.transform_location(location)
        self.transform_location(location)
        if self.overwrite:
            return self.storeind[locationIndex][ind]
        else:
            return self.storeind[locationIndex][ind] != 0

    def numentries(self, ind, location):
        locationIndex = self.transform_location(location)
        if self.overwrite:
            raise Exception("Overwrite flag is present; this function cannot be called")
        return self.storeind[locationIndex][ind]

    # returns tuples of all data available
    def alldata(self, location):
        locationIndex = self.transform_location(location)
        inds = list(self.storeind[locationIndex].keys())
        return inds

    # returns tuples of all data available for a specific independent variable
    # if you want index for example 3rd independent variable's 0s, input (2, 0)
    # 0 - indexed
    def ivardata(self, ivar, location):
        allinds = self.alldata(location)
        return [elem for elem in allinds if elem[ivar[0]] == ivar[1]]

# Index_storage class is an object that stores indices in sparse-matrix format
# it's optimized for the needs of BioActive software in that it encompasses possibilities
# of multiple datatypes from an experiment, such as Raw Data, Derived Data, and Dependent Variables.
# The rows corresponds to the numerical representation of independent variable tuple
# The columns corresponds to the variable type, which is specified in locations attribute of
# this object.
class Index_storage:
    def __init__(self, dim_arr, der, dvars, avars, overwrite):
        self.dim_arr = dim_arr
        self.vars = len(dim_arr)
        self.overwrite = overwrite
        if overwrite:
            self.storeind = sparse.lil_matrix((np.prod(dim_arr), 1 + der + dvars + avars), dtype = np.dtype("?"))
        else:
            self.storeind = sparse.lil_matrix((np.prod(dim_arr), 1 + der + dvars + avars), dtype = np.dtype(np.uint8))

        self.locations = ["Raw Data"]
        for i in range(der):
            self.locations.append("Derived Data " + str(i + 1))
        for i in range(dvars):
            self.locations.append("Dependent Variable " + str(i + 1))
        for i in range(avars):
            self.locations.append("Associated Variable " + str(i + 1))

    # transforms a numpy indices into integer index of listTuples
    def transform_ind(self, ind):
        total = 0
        for i in range(self.vars):
            total += ind[i] * np.prod(self.dim_arr[i+1:])
        return int(total)

    # transforms an integer index of listTuples into indices of numpy array
    def transform_num(self, num):
        base = []
        for i in range(self.vars):
            ind = int(num / np.prod(self.dim_arr[i+1:]))
            num -= ind * np.prod(self.dim_arr[i+1:])
            base.append(ind)
        return tuple(base)

    def transform_location(self, location):
        return self.locations.index(location)

    def store_ind(self, ind, location):
        if self.overwrite:
            self.storeind[self.transform_ind(ind), self.transform_location(location)] = True
        else:
            self.storeind[self.transform_ind(ind), self.transform_location(location)] += 1

    def store_inds(self, inds, location = "Raw Data"):
        for ind in inds:
            self.store_ind(ind, location)

    # returns number of elements that's been stored
    def numelems(self, location = "Raw Data"):
        return self.storeind[..., self.transform_location(location)].count_nonzero()

    # returns true if a data exists in that location
    def hasdata(self, ind, location):
        if self.overwrite:
            return self.storeind[self.transform_ind(ind), self.transform_location(location)]
        else:
            return self.storeind[self.transform_ind(ind), self.transform_location(location)] != 0

    def numentries(self, ind, location):
        if self.overwrite:
            raise Exception("Overwrite flag is present; this function cannot be called")
        return self.storeind[self.transform_ind(ind), self.transform_location(location)]

    # returns tuples of all data available
    def alldata(self, location):
        inds = self.storeind[..., self.transform_location(location)].nonzero()
        return [self.transform_num(ind) for ind in inds[0]]

    # returns tuples of all data available for a specific independent variable
    # if you want index for example 3rd independent variable's 0s, input (2, 0)
    # 0 - indexed
    def ivardata(self, ivar, location):
        allinds = self.alldata(location)
        return [elem for elem in allinds if elem[ivar[0]] == ivar[1]]

# Database object acts as a mediator between the temporary storage space and permanent
# HDF5 strorage, and is optimized for the BioActive software. The temporary storage component
# is handled through self.temp variable which is an temporary storage object, and permanent
# storage is handled through self.f which is a pointer to the HDF5 file on the disc.
# Database object has multiple functions that links between the temporary space and the
# permanent space, along with other functions to relate with CampaignObject which is explained
# in detail in README_Database.md
# The user can essentially treat the Database object as a black-box, but more explanations on how
# the data is passed around is detailed in the comments within this script.

class Database:

    # 0. Initialization
    def __init__(self, dim_arr: object, filename: object, mode: object = 0, der: object = 0,
                dvars: object = 0, avars: object = 0, reset: object = False, sparse: object = True,
                ask: object = False, overwrite: object = True, continuous: object = False,  proactive: object = False) -> object:
        # HDF5 attributes (filename, flag, f)
        if ask:
            name = input("Type the name of the database: ")
            filename = name
        self.filename = filename + ".hdf5"

        # add a flag if the file already exists; pick up from where we left off
        if path.exists(self.filename):
            self.flag = True
        else:
            self.flag = False

        self.f = h5py.File(self.filename,'a')

        # basic attributes (dim_arr, vars, mode, der, dvars, avars, locations, current, overwrite, maxentry)
        self.dim_arr = dim_arr # range of ivars
        self.vars = len(dim_arr) # number of ivars
        self.mode = mode
        self.der = der
        self.dvars = dvars
        self.avars = avars
        self.locations = ["Raw Data"]
        self.proactive = proactive
        for i in range(der):
            self.locations.append("Derived Data " + str(i + 1))
        for i in range(dvars):
            self.locations.append("Dependent Variable " + str(i + 1))
        for i in range(avars):
            self.locations.append("Associated Variable " + str(i + 1))
        if "History" not in self.f:
            self.current = 0
        else:
            self.current = self.getcurrent()
        self.dtypes = {}
        self.overwrite = overwrite
        if not overwrite:
            self.maxentry = {}
            for location in self.locations:
                self.maxentry[location] = 0

        # delete all existing data if reset tag is true
        if reset:
            keys = self.f.keys()
            for link in keys:
                del self.f[link]

        if overwrite:
            tup = tuple(dim_arr)
            if dvars != 0 and not "Dependent Variable" in self.f:
                self.f.create_group("Dependent Variable")
            if der != 0 and not "Derived Data" in self.f:
                self.f.create_group("Derived Data")
            if avars != 0 and not "Associated Variable" in self.f:
                self.f.create_group("Associated Variable")

        # temporary database attributes (temp)
        if sparse:
            self.temp = TempDB_Sparse(dim_arr, der = der, dvars = dvars, avars = avars, 
                    overwrite = overwrite, continuous = continuous)
        else:
            self.temp = TempDB(dim_arr, der = der, dvars = dvars, 
                    avars = avars, overwrite = overwrite)

        Database.storeind = property(lambda self: self.temp.tempind.alldata("Raw Data"))

        # memorizes indices where the actual data is stored
        if "Indices" not in self.f:
            if overwrite:
                self.f.create_dataset("Indices", tuple(dim_arr) + (1 + der + dvars + avars,), dtype = np.bool)
            else:
                self.f.create_dataset("Indices", tuple(dim_arr) + (1 + der + dvars + avars,), dtype = np.uint8)

        self.f.close()

    # 1. Helper Functions

    # transforms a numpy indices into integer index of listTuples
    def transform_ind(self, ind):
        total = 0
        for i in range(self.vars):
            total += ind[i] * np.prod(self.dim_arr[i+1:])
        return int(total)

    # transforms an integer index of listTuples into indices of numpy array
    def transform_num(self, num):
        base = []
        for i in range(self.vars):
            ind = int(num / np.prod(self.dim_arr[i+1:]))
            num -= ind * np.prod(self.dim_arr[i+1:])
            base.append(ind)
        return tuple(base)

    # 2. Linkage between TempDB and HDF5

    # helper function that signals when the flush function should be called
    # called when the user calls store
    # idea is that when storing data into the temporary space, we would not want
    # to use too much temporary space, so when this flush condition is satisfied,
    # Database will automatically flush data into the permanent space.
    def flushcondition(self):
        # the commented code below returns true when temporary database has
        # size above 10MB
        ###################### return get_size(self.temp) > 10
        return False

    # writes all data in temporary database into HDF5 permanent storage
    # data is still present in the temporary space until the program terminates
    def flush(self):
        for destination in self.locations:
        # storage to HDF5
            if self.overwrite:
                self.flush_overwrite(destination)
            else:
                self.flush_nonoverwrite(destination)

        self.temp.reset()

    # helper function that flushes all data when overwrite tag is true
    def flush_overwrite(self, destination):
        f = h5py.File(self.filename, "r+")

        ind = self.locations.index(destination)

        # variable for tempDB
        location = destination

        # pathway formatting
        if destination[:12] == "Derived Data":
            destination = "/Derived Data/" + destination[13:]
        elif destination[:18] == "Dependent Variable":
            destination = "/Dependent Variable/" + destination[19:]
        elif destination[:19] == "Associated Variable":
            destination = "/Associated Variable/" + destination[20:]

        # hardcoded due to limitation of indexing in h5py (v. 2.9.0)
        if self.vars == 2:
            for i1, i2 in self.temp.resetind.alldata(location):
                # possible bug fix--only temporary or only should be temporary
                try:
                    f[destination][i1, i2] = self.temp.show([i1, i2], location)
                except:
                    f[destination][i1, i2] = [self.temp.show([i1, i2], location)]
                f["Indices"][i1, i2, ind] = True

        elif self.vars == 3:
            for i1, i2, i3 in self.temp.resetind.alldata(location):
                f[destination][i1, i2, i3] = self.temp.show([i1, i2, i3], location)
                f["Indices"][i1, i2, i3, ind] = True

        elif self.vars == 4:
            for i1, i2, i3, i4 in self.temp.resetind.alldata(location):
                f[destination][i1, i2, i3, i4] = \
                        self.temp.show([i1, i2, i3, i4], location)
                f["Indices"][i1, i2, i3, i4, ind] = True

        elif self.vars == 5:
            for a, b, c, d, e in self.temp.resetind.alldata(location):
                f[destination][a, b, c, d, e] = \
                        self.temp.show([a, b, c, d, e], location)
                f["Indices"][a, b, c, d, e, ind] = True

        elif self.vars == 6:
            for a, b, c, d, e, f0 in self.temp.resetind.alldata(location):
                f[destination][a, b, c, d, e, f0] = \
                        self.temp.show([a, b, c, d, e, f0], location)
                f["Indices"][a, b, c, d, e, f0, ind] = True

        elif self.vars == 7:
            for a, b, c, d, e, f0, g in self.temp.resetind.alldata(location):
                f[destination][a, b, c, d, e, f0, g] = \
                        self.temp.show([a, b, c, d, e, f0, g], location)
                f["Indices"][a, b, c, d, e, f0, g, ind] = True

        elif self.vars == 8:
            for a, b, c, d, e, f0, g, h in self.temp.resetind.alldata(location):
                f[destination][a, b, c, d, e, f0, g, h] = \
                        self.temp.show([a, b, c, d, e, f0, g, h], location)
                f["Indices"][a, b, c, d, e, f0, g, h, ind] = True

        elif self.vars == 9:
            for a, b, c, d, e, f0, g, h, i in self.temp.resetind.alldata(location):
                f[destination][a, b, c, d, e, f0, g, h, i] = \
                        self.temp.show([a, b, c, d, e, f0, g, h, i], location)
                f["Indices"][a, b, c, d, e, f0, g, h, i, ind] = True

        elif self.vars == 10:
            for a, b, c, d, e, f0, g, h, i, j in self.temp.resetind.alldata(location):
                f[destination][a, b, c, d, e, f0, g, h, i, j] = \
                        self.temp.show([a, b, c, d, e, f0, g, h, i, j], location)
                f["Indices"][a, b, c, d, e, f0, g, h, i, j, ind] = True

        f.close()

    # helper function that flushes all data when overwrite tag is false
    def flush_nonoverwrite(self, destination):
        f = h5py.File(self.filename, "r+")

        ind = self.locations.index(destination)

        # variable for tempDB
        location = destination

        # pathway formatting
        if destination[:12] == "Derived Data":
            destination = "/Derived Data/" + destination[13:]
        elif destination[:18] == "Dependent Variable":
            destination = "/Dependent Variable/" + destination[19:]

        if self.vars == 2:
            for i1, i2 in self.temp.resetind.alldata(location):
                offset = self.temp.tempind.numentries((i1, i2), location) - self.temp.resetind.numentries((i1, i2),
                        location)
                data = self.temp.show([i1, i2], location)[offset:]
                for datum in data:
                    # possible bug fix--only temporary or only should be temporary
                    try:
                        f["/Entry number " + str(offset + 1) + "/" + destination][i1, i2] = datum
                    except:
                        f["/Entry number " + str(offset + 1) + "/" + destination][i1, i2] = [datum]
                    offset += 1
                    f["Indices"][i1, i2, ind] += 1

        elif self.vars == 3:
            for i1, i2, i3 in self.temp.resetind.alldata(location):
                offset = self.temp.tempind.numentries((i1, i2, i3), location) - \
                        self.temp.resetind.numentries((i1, i2, i3), location)
                data = self.temp.show([i1, i2, i3], location)[offset:]
                for datum in data:
                    f["/Entry number " + str(offset + 1) + "/" + destination][i1, i2, i3] = datum
                    offset += 1
                    f["Indices"][i1, i2, i3, ind] += 1

        elif self.vars == 4:
            for i1, i2, i3, i4 in self.temp.resetind.alldata(location):
                offset = self.temp.tempind.numentries((i1, i2, i3, i4), location) -\
                        self.temp.resetind.numentries((i1, i2, i3, i4), location)
                data = self.temp.show([i1, i2, i3, i4], location)[offset:]
                for datum in data:
                    f["/Entry number " + str(offset + 1) + "/" + destination][i1, i2, i3, i4] = datum
                    offset += 1
                    f["Indices"][i1, i2, i3, i4, ind] += 1

        elif self.vars == 5:
            for a, b, c, d, e in self.temp.resetind.alldata(location):
                offset = self.temp.tempind.numentries((a, b, c, d, e), location) - \
                        self.temp.resetind.numentries((a, b, c, d, e), location)
                data = self.temp.show([a, b, c, d, e], location)[offset:]
                for datum in data:
                    f["/Entry number " + str(offset + 1) + "/" + destination][a, b, c, d, e] = datum
                    offset += 1
                    f["Indices"][a, b, c, d, e, ind] += 1

        elif self.vars == 6:
            for a, b, c, d, e, f0 in self.temp.resetind.alldata(location):
                offset = self.temp.tempind.numentries((a, b, c, d, e, f0), location) - \
                        self.temp.resetind.numentries((a, b, c, d, e, f0), location)
                data = self.temp.show([a, b, c, d, e, f0], location)[offset:]
                for datum in data:
                    f["/Entry number " + str(offset + 1) + "/" + destination][a, b, c, d, e, f0] = datum
                    offset += 1
                    f["Indices"][a, b, c, d, e, f0, ind] += 1

        elif self.vars == 7:
            for a, b, c, d, e, f0, g in self.temp.resetind.alldata(location):
                offset = self.temp.tempind.numentries((a, b, c, d, e, f0, g), location) - \
                        self.temp.resetind.numentries((a, b, c, d, e, f0, g), location)
                data = self.temp.show([a, b, c, d, e, f0, g], location)[offset:]
                for datum in data:
                    f["/Entry number " + str(offset + 1) + "/" + destination][a, b, c, d, e, f0, g] = datum
                    offset += 1
                    f["Indices"][a, b, c, d, e, f0, g, ind] += 1

        elif self.vars == 8:
            for a, b, c, d, e, f0, g, h in self.temp.resetind.alldata(location):
                offset = self.temp.tempind.numentries((a, b, c, d, e, f0, g, h), location) - \
                        self.temp.resetind.numentries((a, b, c, d, e, f0, g, h), location)
                data = self.temp.show([a, b, c, d, e, f0, g, h], location)[offset:]
                for datum in data:
                    f["/Entry number " + str(offset + 1) + "/" + destination][a, b, c, d, e, f0, g, h] = datum
                    offset += 1
                    f["Indices"][a, b, c, d, e, f0, g, h, ind] += 1

        elif self.vars == 9:
            for a, b, c, d, e, f0, g, h, i in self.temp.resetind.alldata(location):
                offset = self.temp.tempind.numentries((a, b, c, d, e, f0, g, h, i), location) - \
                        self.temp.resetind.numentries((a, b, c, d, e, f0, g, h, i), location)
                data = self.temp.show([a, b, c, d, e, f0, g, h, i], location)[offset:]
                for datum in data:
                    f["/Entry number " + str(offset + 1) + "/" + destination][a, b, c, d, e, f0, g, h, i] = datum
                    offset += 1
                    f["Indices"][a, b, c, d, e, f0, g, h, i, ind] += 1

        elif self.vars == 10:
            for a, b, c, d, e, f0, g, h, i, j in self.temp.resetind.alldata(location):
                offset = self.temp.tempind.numentries((a, b, c, d, e, f0, g, h, i, j), location) - \
                        self.temp.resetind.numentries((a, b, c, d, e, f0, g, h, i, j), location)
                data = self.temp.show([a, b, c, d, e, f0, g, h, i, j], location)[offset:]
                for datum in data:
                    f["/Entry number " + str(offset + 1) + "/" + destination][a, b, c, d, e, f0, g, h, i, j] = datum
                    offset += 1
                    f["Indices"][a, b, c, d, e, f0, g, h, i, j, ind] += 1

        f.close()

	# 3. Main functionalities (storage and retrieval)

	# store function
	# REQUIRES: array of tuples (data, i1, ..., in) in which i represent the
	#           indices of independent variables for n-dimensional experimental space
	#           if images were to be passed in, the data part will be string of image file
	# ENSURES: data is stored in temporary database. Nothing's returned
	# Data is stored first inside the temporary space, and is only moved to the
	# disc when flush is called.
 
    def store(self, arr, destination = "Raw Data"):
        f = h5py.File(self.filename,'r+')
        # automatic flushing mechanism
        if self.flushcondition():
            self.flush()

        # dataset is not initialized with the database
        # instead, the store function will detect the datatype of the first data
        # being passed in, and initialize the dataset according to that type.
        # It is IMPERATIVE to keep the datatype the same within the same "destination"

        if destination not in self.dtypes.keys():
            self.dtypes[destination] = dtype_extraction(arr[0][0])

        if self.requirenew(arr[0], destination):
            self.makenewdset(destination)

        for elem in arr:
            if self.mode == 1 and destination == "Raw Data":
                self.image_store(elem, destination)
            else:
                self.local_store(elem, destination)

        f.close()

    # helper function that stores data in temporary storage object
    def local_store(self, tup, destination):
        self.temp.write(tup, destination)

    # helper function that stores images as binary of the image file
    def image_store(self, tup, destination):
        data = open(tup[0], "rb").read()
        data = np.fromstring(data, dtype = "uint8")
        self.temp.write((data,) + tup[1:], destination)

    # helper function that checks if new dataset should be initialized
    def requirenew(self, elem, destination):
        f = h5py.File(self.filename,'r+')

        if self.overwrite:
            if destination == "Raw Data":
                res = destination not in f
            elif destination[:12] == "Derived Data":
                res = destination[13:] not in f["Derived Data"]
            elif destination[:18] == "Dependent Variable":
                res = destination[19:] not in f["Dependent Variable"]
            elif destination[:19] == "Associated Variable":
                res = destination[20:] not in f["Associated Variable"]
        else:
            res = self.temp.tempind.numentries(elem[1:], destination) == self.maxentry[destination]

        f.close()
        return res

    # helper function that creates appropriate dataset if it doesn't exist already
    def makenewdset(self, destination):
        f = h5py.File(self.filename,'r+')
        if self.overwrite:
            if destination == "Raw Data":
                if self.mode == 0:
                    dt = self.dtypes[destination]
                elif self.mode == 1:
                    dt = h5py.special_dtype(vlen = np.dtype("uint8"))
                f.create_dataset("Raw Data", tuple(self.dim_arr),\
                        dtype = dt)
            elif destination[:12] == "Derived Data":
                f["Derived Data"].create_dataset(destination[13:],\
                        tuple(self.dim_arr), dtype = self.dtypes[destination])
            elif destination[:18] == "Dependent Variable":
                f["Dependent Variable"].create_dataset(destination[19:],\
                        tuple(self.dim_arr), dtype = self.dtypes[destination])
        else:
            self.maxentry[destination] += 1
            entrynum = self.maxentry[destination]
            name = "Entry number " + str(entrynum)

            if name not in f:
                f.create_group(name)
                if self.der != 0:
                    f[name].create_group("Derived Data")
                if self.dvars != 0:
                    f[name].create_group("Dependent Variable")
                if self.avars != 0:
                    f[name].create_group("Associated Variable")

            if destination == "Raw Data":
                f[name].create_dataset("Raw Data", tuple(self.dim_arr),\
                        dtype = self.dtypes[destination])
            elif destination[:12] == "Derived Data":
                f["/" + name + "/Derived Data"].create_dataset(destination[13:],\
                        tuple(self.dim_arr), dtype = self.dtypes[destination])
            elif destination[:18] == "Dependent Variable":
                f["/" + name + "/Dependent Variable"].create_dataset(destination[19:],\
                        tuple(self.dim_arr), dtype = self.dtypes[destination])

        f.close()

    # retrieve function
    # input takes different meaning depending on the flag. Refer to README_Database.md
    # data is read from the temporary space if archived == False, and from the HDF5
    # if archived == True, assuming that the data has been flushed previously.

    # flag = "one" just return one element
    # flag = "all" returns every data we have
    # flag = "arr" returns all elements for the indices within the array
    # flag = "ivar" returns all elements for a specific element of a specific independent variable
    # flag = "entry" returns all data in related entry as a tuple
    def retrieve(self, getfrom, location = "Raw Data", archived = False, flag = "one", entry = -1):
        if flag == "one":
            res = self.retrieve_one(getfrom, location, archived)
        elif flag == "arr":
            result = []
            for ind in getfrom:
                result.append((self.retrieve(ind, location,
                    archived, flag = "one", entry = entry),) + ind)
            res = result
        elif flag == "all":
            res = self.retrieve(self.temp.tempind.alldata(location),
                    location, archived, flag = "arr", entry = entry)
        elif flag == "ivar":
            res = self.retrieve(self.temp.tempind.ivardata(getfrom, location),
                    location, archived, flag = "arr", entry = entry)
        elif flag == "entry":
            res = (self.retrieve(getfrom, location = "Raw Data", archived = archived,
                flag = "one", entry = entry),)
            for location in self.locations[1:]:
                res += (self.retrieve(getfrom, location = location,
                    archived = archived, flag = "one", entry = entry),)
        else:
            raise Exception("Wrong flag ... the options are one, arr, all, ivar, and entry")

        if entry != -1 and self.overwrite:
            raise Exception("Cannot manipulate entry variable in overwrite mode")

        elif entry != -1 and flag == "one":
            try:
                res = res[entry - 1]
            except IndexError:
                print("Requested entry does not exist for that experiment. The highest entry for that experiment is %s" % \
                    self.temp.tempind.numentries(getfrom, location))
                res = np.nan

        if self.proactive and flag == "one" and location == "Raw Data":
            # Calculates the confidence of a given label based on its experimental history.  Assumes that an experiment
            # with confidence c has a c% chance of being correct and (1-c)% chance of being a random incorrect label
            # TODO Currently hard coding dcarCategories for UberSL.  Need a way to get from campaignObject
            dVarCategories = [0, 1]
            # retrieves the history of experiment confidences
            confidenceData = self.retrieve(getfrom, location="Derived Data 2", flag='one')
            # retrieves the results of the experiments
            experimentResults = self.retrieve_one(getfrom, location, archived)

            # Set up a likelihood entry for every possible result
            likelihoods = [0] * len(dVarCategories)
            for ind in range(len(likelihoods)):
                likelihood = 1
                for expInd in range(len(experimentResults)):
                    experimentRes = experimentResults[expInd]
                    # If experimental result matches the outcome whose likelihood we are calculating,
                    # multiply by confidence
                    if experimentRes == dVarCategories[ind]:
                        likelihood *= confidenceData[expInd]
                    else:
                        #  Otherwise multiply by probability of randomly selecting result from the remaining choices
                        likelihood *= (1 - confidenceData[expInd]) / (len(dVarCategories) - 1)
                likelihoods[ind] = likelihood
            # return the result of the most likely label, as well as the likelihood
            labelConfidence = np.max(likelihoods) / np.sum(likelihoods)
            res = (dVarCategories[np.argmax(likelihoods)], labelConfidence)

        return res

    # returns the data at the given index
    # will raise error if the requested data doesn't exist
    def retrieve_one(self, ind, location, archived):
        if not self.temp.exists(ind, location):
            raise DataDoesNotExistException("Data doesn't exist inside the database")

        # assumed that when raw data is image, we would like to work with the first
        # dependent variable instead
        if self.mode == 1 and location == "Raw Data":
            location = "Dependent Variable 1"

        if not archived:
            data = self.temp.show(ind, location)
        elif self.overwrite:
            f = h5py.File(self.filename, "r")

            # pathway formatting
            if location[:12] == "Derived Data":
                location = "/Derived Data/" + location[13:]
            elif location[:18] == "Dependent Variable":
                location = "/Dependent Variable/" + location[19:]
            data = f[location]

            for idx in ind:
                data = data[idx]

            f.close()

        else:
            f = h5py.File(self.filename, "r")

            maxnum = f.get("Indices")[..., self.locations.index(location)][ind]

            # pathway formatting
            if location[:12] == "Derived Data":
                location = "/Derived Data/" + location[13:]
            elif location[:18] == "Dependent Variable":
                location = "/Dependent Variable/" + location[19:]

            data = []
            for i in range(maxnum):
                datatemp = f["/Entry number " + str(i + 1) + "/" + location]
                for idx in ind:
                    datatemp = datatemp[idx]
                data.append(datatemp)

            f.close()

        if self.mode == 1 and location == "Raw Data":
            data = Image.open(io.BytesIO(data))
        return data

    # 4. Campaign Object compatibility
    # the codes in this section is specific to BattleshipReal campaign and will
    # need campaign object to have breakpoints functions

    # The group "CampaignObject" in the root group of the HDF5 file is analogous
    # to the campaignObject in BioActive; the HDF5 version only saves the historical
    # attributes such as ESC, accuracy, etc. Other attributes are re-initialized
    # when the campaign is run again
    def campaignflush(self, campaignObject):
        self.flush()
        campaignObject.batch -= 1

        nosavelist = ["data", "ESS", "plotting", "goalTether", "modelData",\
                "activeLearner", "fetchData", "breakpoint", "xs", "ys"]
        returnaslist = []
        returnasdict = []
        for label in dir(campaignObject):
            if label not in nosavelist and label[0] != "_":
                data = getattr(campaignObject, label)
                self.addattr(label, data)
                # memorize attributes that has to come out as list
                if isinstance(data, list):
                    returnaslist.append(label)
                if isinstance(data, dict):
                    returnasdict.append(label)

        self.addattr("returnaslist", returnaslist)
        self.addattr("returnasdict", returnasdict)

        campaignObject.batch += 1

    # add attributes to the group "CampaignObject" to save attributes from the
    # campaignObject
    def addattr(self, label, item):
        f = h5py.File(self.filename, "r+")

        if "CampaignObject" not in f:
                f.create_group("CampaignObject")

        grp = f["CampaignObject"]
        if label == "model":
            # dimmaj (0 and 1)
            for i in range(len(item)):
                for j, grouping in enumerate(item[i]):
                    newlb = "model" + str(i) + "_" + str(j)
                    grp.attrs[newlb] = grouping
        elif isinstance(item, dict):
            grp.attrs[label] = str(item)
        else:
            grp.attrs[label] = item
        f.close()

    # load back all the data in HDF5 into temporary database
    # also load back all the attributes to campaignObject
    def load(self, campaignObject):
        f = h5py.File(self.filename, "r")

        # loading indices into temporary space
        res = []
        for i in range(len(self.locations)):
            indmat = f.get("Indices")[..., i].flatten()
            res.append(indmat)

        self.temp.tempind.storeind = sparse.lil_matrix(np.transpose(res))

        # loading data into temporary space
        for location in self.locations:
            for ind in self.temp.tempind.alldata(location):
                data = self.retrieve(ind, location, archived = True)
                tup = (data,) + ind
                self.temp.write(tup, location)

        # loading object attributes
        grp = f["CampaignObject"]
        returnaslist = grp.attrs["returnaslist"]
        returnasdict = grp.attrs["returnasdict"]
        model = []
        for label in grp.attrs.keys():
            if label in returnasdict:
                import ast
                data = grp.attrs[label]
                setattr(campaignObject, label, ast.literal_eval(data))
            elif len(label) >= 5 and label[:5] == "model":
                dimmaj = int(label[5])
                if len(model) < dimmaj + 1:
                    model.append([])
                model[dimmaj].append(grp.attrs[label])
            elif label != "returnaslist":
                data = grp.attrs[label]
                if label in returnaslist:
                    data = data.tolist()
                setattr(campaignObject, label, data)
        setattr(campaignObject, "model", model)

        dataRequested = f["CampaignObject"].attrs["lastDataRequest"]
        f.close()

        # assume the requested experiment is terminated and requests the
        # generated data to pick up the campaign from where the user left it
        platedata = campaignObject.breakpoint(campaignObject).perform()

        neededData = []
        for ind in dataRequested:
            data = platedata[ind]
            neededData.append((data,) + self.transform_num(ind))
        self.store(neededData)
        campaignObject.batch += 1

    # checks if the flag (did the HDF5 exist before?) is present, and if so,
    # let the user to decide pick up from previous progress or not
    # based on the intuition that there is a 1:1 relationsip between the HDF5 file
    # and the instance of a campaign
    def flagcheck(self, campaignObject):
        if self.flag:
            while 1:
                try:
                    command = input("The campaign was initialized at %s \n\ and last saved at %s. \n\ Currently at Batch number %s. \n\ BioActive expects the user to perform experiment according to the protocol %s and its results. \n\ Continue from previous progress? [Y/n]" % self.loadhistory())
                except KeyError:
                    # this is when campaignObject is not saved in HDF5, so we just make a new Database
                    campaignObject.data = Database(self.dim_arr, self.filename[:-5])
                    campaignObject.breakpoint(campaignObject).initialize()
                    break
                if command == "Y":
                    self.load(campaignObject)
                    break
                elif command == "n":
                    while 1:
                        try:
                            campaignObject.data = Database(self.dim_arr,
                                    self.filename[:-5], reset = True)
                            break
                        except KeyError:
                            input("HDF5 file might be open. Try closing the file so it can be reset and then hit enter.")
                    campaignObject.breakpoint(campaignObject).initialize()
                    break
        else:
            campaignObject.breakpoint(campaignObject).initialize()

    # helper function that returns information about start time, end time,
    # current batch number, and the most recent protocol
    def loadhistory(self):
        f = h5py.File(self.filename, "r")
        start = f["CampaignObject"].attrs["starttime"]
        end = f["CampaignObject"].attrs["savetime"]
        batch = f["CampaignObject"].attrs["batch"]
        protocol = f["CampaignObject"].attrs["lastprotocol"]
        f.close()
        return (start, end, batch + 1, protocol)


    # 5. Batch History storage

    # stores history of batch of experiments active learner requests
    def storehistory(self, arr):
        f = h5py.File(self.filename, "r+")

        indtype = []
        for i in range(self.vars):
            indtype.append(("index" + str(i + 1), np.intp))
        dt = np.dtype(indtype)

        if self.current == 0:
            f.create_dataset("History", (100, len(arr)),\
            maxshape = (None, len(arr)), dtype = dt, chunks = (1, len(arr)),\
            fillvalue = None)

        currentmax, foo = f["History"].shape
        if self.current >= currentmax:
            f["History"].resize((currentmax + 100, len(arr)))

        for i in range(len(arr)):
            f["History"][self.current, i] = self.transform_num(arr[i])
        
        self.current += 1

        f.close()

    # returns the current batch number assuming "History" is already initialized
    def getcurrent(self):
        f = h5py.File(self.filename, "r")
        x, y = f["History"].shape
        for i in range(x):
            check1 = True
            check2 = True
            for j in range(self.vars):
                check1 = check1 and f["History"][i, 0][j] == 0
                check2 = check2 and f["History"][i, 1][j] == 0
                if not check1 and check2:
                    break
            if check1 and check2:
                return i

        # when the history is already filled with maximum size
        return x + 1

    # returns the history for that specific batch number
    def gethistory(self, batch):
        f = h5py.File(self.filename)

        bat = f["History"][batch]
        f.close()

        return bat

# other helper functions used in Database object
def dtype_extraction(elem):
    if isinstance(elem, (float, np.float_)):
        dt = float
    elif isinstance(elem, (int, np.int_)):
        dt = int
    elif isinstance(elem, str):
        dt = h5py.special_dtype(vlen = str)
    elif isinstance(elem, tuple):
        tp = []
        for i in range(len(elem)):
            tp.append(("Feature " + str(i + 1), dtype_extraction(elem[i])))
        dt = np.dtype(tp)
    elif isinstance(elem, list):
        pseudo_elem = np.asarray(elem)
        dt = dtype_extraction(pseudo_elem)
    elif isinstance(elem, np.ndarray):
        shape = elem.shape
        base = elem
        for i in range(len(shape)):
            base = base[0]
        # figure out the base type
        tp = str(shape).replace(" ", "") + "f8"
        dt = np.dtype(tp)
    else:
        raise Exception("Datatype is not supported")
    return dt

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


class DatabaseException(Exception):
    pass


class DataDoesNotExistException(DatabaseException):
    pass


