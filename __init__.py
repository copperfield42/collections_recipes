"""Collections recipes"""

from __future__ import division

from collections import *
from collections import abc
from itertools import chain as _chain
from operator import attrgetter as _attrgetter, itemgetter as _itemgetter
from contextlib import suppress as _suppress, AbstractContextManager as _AbstractContextManager
import operator as _operator
import sqlite3 as _sqlite3
import json as _json
import pickle as _pickle
import pathlib as _pathlib
from bisect import bisect_left as _bisect_left, bisect_right as _bisect_right

import abc_recipes as _ABC

################################################################################
#----------------------------- Collections Recipes -----------------------------
################################################################################


class DeepChainMap(ChainMap):
    """Variant of ChainMap that allows direct updates to inner scopes.

    The ChainMap class only makes updates (writes and deletions) to the first mapping in the chain while lookups will search the full chain. However, if deep writes and deletions are desired, it is easy to make a subclass that updates keys found deeper in the chain.

    >>> d = DeepChainMap({'zebra': 'black'}, {'elephant': 'blue'}, {'lion': 'yellow'})
    >>> d['lion'] = 'orange'         # update an existing key two levels down
    >>> d['snake'] = 'red'           # new keys get added to the topmost dict
    >>> del d['elephant']            # remove an existing key one level down
    >>> d                            # display result
    DeepChainMap({'zebra': 'black', 'snake': 'red'}, {}, {'lion': 'orange'})
    >>>
    """

    def __setitem__(self, key, value):
        for mapping in self.maps:
            if key in mapping:
                mapping[key] = value
                return
        self.maps[0][key] = value

    def __delitem__(self, key):
        for mapping in self.maps:
            if key in mapping:
                del mapping[key]
                return
        raise KeyError(key)

def tail(filename, n=10):
    'Return the last n lines of a file'
    with open(filename) as f:
        return deque(f, n)

def moving_average(iterable, n=3):
    """moving_average([40, 30, 50, 46, 39, 44]) --> 40.0 42.0 45.0 43.0
       http://en.wikipedia.org/wiki/Moving_average"""
    it = iter(iterable)
    d = deque(itertools.islice(it, n-1))
    d.appendleft(0)
    s = sum(d)
    for elem in it:
        s += elem - d.popleft()
        d.append(elem)
        yield s / n

def delete_nth(d:deque, n:int):
    d.rotate(-n)
    d.popleft()
    d.rotate(n)


class LastUpdatedOrderedDict(OrderedDict):
    'Store items in the order the keys were last added'

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        super().__setitem__(key, value)
        #OrderedDict.__setitem__(self, key, value)

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return "{}([{}])".format( self.__class__.__name__, ", ".join(map(repr,self.items())) )

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class ListBasedSet(abc.Set):
    ''' Alternate set implementation favoring space over speed
        and not requiring the set elements to be hashable. '''
    def __init__(self, iterable):
        self.elements = lst = []
        for value in iterable:
            if value not in lst:
                lst.append(value)

    def __iter__(self):
        return iter(self.elements)

    def __contains__(self, value):
        return value in self.elements

    def __len__(self):
        return len(self.elements)


################################################################################
#------------------------------ Raymond Hettinger ------------------------------
################################################################################
# Examples show by Raymond Hettinger in the video
# https://www.youtube.com/watch?v=S_ipdVNSFlo




class BitSet(abc.MutableSet):
    """Ordered set with compact storage for integers in a fixed range"""

    def __init__(self, limit, iterable=()):
        self.limit = limit
        num_bytes = (limit+7)//8
        self.data = bytearray(num_bytes)
        self |= iterable

    def _get_location(self, elem):
        if elem < 0 or elem>= self.limit:
            raise ValueError(f"{elem!r} must be in range 0 <= elem < {self.limit}")
        return divmod(elem,8)

    def __contains__(self, elem):
        bytenum, bitnum = self._get_location(elem)
        return bool( (self.data[bytenum] >> bitnum) & 1 )

    def add(self, elem):
        bytenum, bitnum = self._get_location(elem)
        self.data[bytenum]  |= ( 1 << bitnum )

    def discard(self, elem):
        bytenum, bitnum = self._get_location(elem)
        self.data[bytenum] &= ~( 1 << bitnum )

    def __iter__(self):
        for elem in range(self.limit):
            if elem in self:
                yield elem

    def __len__(self):
        return sum(1 for elem in self)

    def __repr__(self):
        return f"{type(self).__name__}(limit={self.limit}, iterable={list(self)})"

    def _from_iterable(self, iterable):
        #necesary because the constructor take an extra argument
        #see: Notes on using Set and MutableSet as a mixin
        #in the documentation
        return type(self)(self.limit, iterable)


class SQLDict(abc.MutableMapping):
    """Dictionary-like object with database back-end store.

       Concurrent and persistent.
       Easy to share with other programs
       Queryable
       Single file (easy to email and backup).
       """

    def __init__(self, dbname, items=(), **kwarg):
        self.dbname = dbname
        self.conn = _sqlite3.connect(dbname)
        c = self.conn.cursor()
        with _suppress(_sqlite3.OperationalError):
            c.execute( "CREATE TABLE Dict (key text, value text)" )
            c.execute( "CREATE UNIQUE INDEX kndx ON Dict (key)" )
        self.update(items, **kwarg)

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        with self.conn as c:
            c.execute( "INSERT INTO Dict VALUES (?,?)", (key, value) )

    def __getitem__(self, key):
        c = self.conn.execute( "SELECT value FROM Dict WHERE key=?", (key,) )
        row = c.fetchone()
        if row is None:
            raise KeyError(key)
        return row[0]

    def __delitem__(self,key):
        if key not in self:
            raise KeyError(key)
        with self.conn as c:
            c.execute( "DELETE FROM Dict WHERE key=?", (key,) )

    def __len__(self):
        return next(self.conn.execute("SELECT COUNT(*) FROM Dict"))[0]

    def __iter__(self):
        c = self.conn.execute("SELECT key FROM Dict")
        return map(_itemgetter(0), c.fetchall())

    def __repr__(self):
        return f"{type(self).__name__}(dbname={self.dbname!r}, items={list(self.items())}"

    def close(self):
        self.conn.close()



################################################################################
#--------------------------------- Mis Recipes ---------------------------------
################################################################################

#TO DO: añadir el resto de funciones que tiene set

class OrderedSet(abc.MutableSet):
    """Set that remember the order of insertion"""

    __slots__ = ("_elements")

    def __init__(self, iterable=None):
        if iterable is None:
            self._elements = OrderedDict()
        else:
            self._elements = OrderedDict.fromkeys(iterable)

    def __contains__(self, value):
        return value in self._elements

    def __iter__(self):
        return iter(self._elements)

    def __len__(self):
        return len(self._elements)

    def __repr__(self):
        return "{}([{}])".format(self.__class__.__name__, ", ".join(map(repr,self)) )

    def add(self, value):
        self._elements[value] = None

    def discard(self, value):
        try:
            del self._elements[value]
        except KeyError:
            pass

    def clear(self):
        """Remove all elements in-place"""
        self._elements.clear()

    def move_to_end(self, value, last=True):
        """Move an existing value to either end of an ordered set.
           The item is moved to the right end if last is true (the default)
           or to the beginning if last is false.
           Raises KeyError if the value does not exist"""
        self._elements.move_to_end(value, last)

    def copy(self):
        return self.__class__(self)

    def difference(self, *other):
        new = self.copy()
        for elem in other:
            if not new:
                break
            new -= elem
        return new

    def difference_update(self, other):
        raise NotImplementedError

    def intersection(self, *other):
        new = self.copy()
        for elem in other:
            if not new:
                break
            new &= elem
        return new

    def intersection_update(self, other):
        raise NotImplementedError

    def issubset(self, other):
        return self <= set(other)

    def issuperset(self, other):
        return self >= set(other)

    def symmetric_difference(self, other):
        return self ^ other

    def symmetric_difference_update(self, other):
        raise NotImplementedError

    def union(self, *other):
        new = self.copy()
        for elem in other:
            new |= elem
        return new

    def update(self, other):
        raise NotImplementedError


_get2 = _operator.attrgetter(*("start stop".split()) )
_get3 = _operator.attrgetter(*("start stop step".split()) )
_basestring = (str,bytes)

class chr_range(abc.Sequence):
    #http://stackoverflow.com/q/30362799/5644961S

    def __init__(self,*argv):
        sl = slice(*argv)
        argv = [ x or y for x,y in zip(_get3(sl),('\x00',None,1)) ]
        if not all(argv):
            raise ValueError
        if not all( isinstance(x,t) for x,t in zip(argv,[str,str,int]) ):
            raise TypeError
        #print(argv)
        start, stop = map(ord,argv[:2])
        self._range = range(start,stop, argv[-1])
        for a,v in zip("start stop step".split(),argv):
            setattr(self,a,v)

    def __repr__(self):
        v = _get2(self) if self.step == 1 else _get3(self)
        return "{}({})".format(self.__class__.__name__,  ", ".join(map("{!r}".format,v)))

    def __iter__(self):
        return map(chr, self._range)

    def __reversed__(self):
        return map(chr, reversed(self._range))

    def __contains__(self,key):
        return ord(key) in self._range

    def __getitem__(self, index):
        if isinstance(index,slice):
            new_range = self._range[index]
            start, stop = map(chr,_get2(new_range))
            return self.__class__(start, stop, new_range.step)
        return chr( self._range[index] )

    def __len__(self):
        return len(self._range)

    def index(self, value, *argv, **kwarg):
        return self._range.index( ord(value), *argv, **kwarg)

    def count(self, values):
        return int( value in self )




class SortedSequence(abc.MutableSequence):
    """Sequence that keep its elements ordered"""
    #https://code.activestate.com/recipes/577197-sortedcollection/

    def _getkey(self):
        return self._key

    def _setkey(self, key):
        if key is not self._key:
            self.__init__(self._items, key=key)

    def _delkey(self):
        self._setkey(None)

    key = property(_getkey, _setkey, _delkey, 'key function')

    def __init__(self, iterable=(), *, key=None):
        self._given_key = key
        key = (lambda x:x) if key is None else key
        sortpairs   = sorted(((key(v),v) for v in iterable), key=_itemgetter(0))
        self._keys  = list(map(_itemgetter(0),sortpairs))
        self._items = list(map(_itemgetter(1),sortpairs))
        self._key   = key

    def __getitem__(self, index):
        return self._items[index]

    def __len__(self):
        return len(self._items)

    def __delitem__(self, index):
        del self._items[index]
        del self._keys[index]

    def __contains__(self, item):
        k = self._key(item)
        lo = _bisect_left(self._keys,k)
        hi = _bisect_right(self._keys,k)
        items = self._items
        return any( item == items[i] for i in range(lo,hi))

    def __iter__(self):
        return iter(self._items)

    def __reversed__(self):
        return reversed(self._items)

    def __repr__(self):
        if self._given_key:
            return f"{type(self).__name__}({self._items!r}, key={getattr(self._given_key, '__qualname__', repr(self._given_key))})"
        return f"{type(self).__name__}({self._items!r})"

    def insert(self, item):
        'Insert a new item.  If equal keys are found, add to the left'
        k = self._key(item)
        i = _bisect_left(self._keys, k)
        self._keys.insert(i,k)
        self._items.insert(i,item)

    insert_left = insert

    def insert_right(self, item):
        'Insert a new item.  If equal keys are found, add to the right'
        k = self._key(item)
        i = _bisect_right(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, item)

    append = insert_right

    def __setitem__(self,index,item):
        """this does not support item assignment"""
        raise TypeError(f"{type(self).__name__!r} object does not support item assignment")
        k = self._key(item)
        i = _bisect_left(self._keys, k)
        j = _bisect_right(self._keys, k)
        print(f"{item=} {index=} {i=} {j=}")
        if (i==j ) and i == index:
            self._keys[i] =  k
            self._items[i] = item
        else:
            raise ValueError("Can't insert item in position {index} and preserve the order")




class LineSeekableFile(abc.Sequence,_AbstractContextManager):
    #https://stackoverflow.com/a/59185917/5644961

    def __init__(self, filepath:str, linesoffset:abc.Sequence=None):
        self._filepath = filepath
        self._file = file = open(filepath)
        if linesoffset:
            self.linesoffset = linesoffset
        else:
            self.linesoffset = lst =[0,*( file.tell() for _ in iter(file.readline,"")) ]
            lst.pop()

    def close(self):
        self._file.close()

    def __repr__(self):
        return f"{type(self).__name__}({self._filepath!r})"

    def __len__(self):
        """Number of lines of this file"""
        return len(self.linesoffset)

    def __getitem__(self,index):
        self._file.seek(self.linesoffset[index])
        return self._file.readline()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()










####################################################
#-------------------- File Dict --------------------
####################################################


class ValueMapDict(abc.MutableMapping):
    """Coperative class that apply a transformation to the store value
       before deliveriong/store it

       Example:
       >>> class LenValueDict(ValueMapDict, collections.UserDict):
               "dict that only store the first n elements of value"
               def __init__(self, valuelen, *arg, **karg):
                   super().__init__(*arg, **karg)
                   self.valuelen = valuelen
               def valuemapwrite(self, value):
                   return value[:self.valuelen]
               def valuemapread(self, value):
                   return value


       >>> a = LenValueDict(3)
       >>> a[1] = [1,2,3,4,5,6]
       >>> a[1]
       [1, 2, 3]
       >>>
       """

    @_ABC.abstractmethod
    def valuemapwrite(self,value):
        """Transformation function for the value to store it"""
        return value

    @_ABC.abstractmethod
    def valuemapread(self,value):
        """Transformation function for the value to read it"""
        return value

    def __getitem__(self,key):
        return self.valuemapread( super().__getitem__(key) )

    def __setitem__(self,key,value):
        super().__setitem__(key, self.valuemapwrite(value) )

class KeyMapDict(abc.MutableMapping):
    """Coperative class that apply a transformation to the key
       before store/consult it

       the __iter__ method should apply the inverse of keymap if nessesary

       Example:
       >>> class CaseInsensitiveDict(KeyMapDict, collections.UserDict):
               def keymap(self,key:str):
                     return key.lower()


       >>> a = CaseInsensitiveDict()
       >>> a["fun"] = 1
       >>> "FUN" in a
       True
       >>> a
       {'fun': 1}
       >>>


       """

    @_ABC.abstractmethod
    def keymap(self,key):
        """Transformation function for the key to store/consult"""
        return key

    def __getitem__(self,key):
        return super().__getitem__( self.keymap(key) )

    def __setitem__(self,key,value):
        super().__setitem__( self.keymap(key), value )

    def __delitem__(self,key):
        super().__delitem__( self.keymap(key) )

    def __contains__(self,key):
        return super().__contains__( self.keymap(key) )

class FileDict(abc.MutableMapping, _ABC.ConfigClass):
    """File based dictionary

       A dictionary-like object based on the file system rather than
       in-memory hash tables. It is persistent and sharable between
       proceses

       This is a variation on Raymond Hettinger's original idea
       as presented in this video:
       https://www.youtube.com/watch?v=S_ipdVNSFlo
       """

    def __init__(self, dirpath:str, data=(), *, text_mode:bool=True, encoding:str="utf8", errors:str=None):
        dirpath = _pathlib.Path(dirpath)
        dirpath.mkdir(exist_ok=True)
        self._dirpath   = dirpath
        self.text_mode  = text_mode
        self.encoding   = encoding
        self.errors     = errors
        self.update(data)

    @property
    def dirpath(self) -> _pathlib.Path:
        """Folder to hold this dict data"""
        return self._dirpath

    @dirpath.setter
    def dirpath(self, path:str):
        config = self.config
        config["dirpath"] = path
        self.__init__(**config)

    @property
    def text_mode(self) -> bool:
        """Determine if the files are read in text or byte mode"""
        return self._text_mode

    @text_mode.setter
    def text_mode(self, value:bool):
        self._text_mode = bool(value)

    @property
    def encoding(self) -> str:
        """same meaning as the encoding optional argument of open"""
        return self._encoding

    @encoding.setter
    def encoding(self,value:str):
        self._encoding = value

    @property
    def errors(self) -> str:
        """same meaning as the errors optional argument of open"""
        return self._errors

    @errors.setter
    def errors(self, value:str):
        self._errors = value

    @property
    def config(self) -> dict:
        """configuration of this dict"""
        config = super().config
        config.update( dirpath = self.dirpath, text_mode = self.text_mode )
        if self.text_mode:
            config.update( encoding = self.encoding, errors = self.errors )
        return config

    def __getitem__(self, key):
        try:
            path = self.dirpath / key
            if self.text_mode:
                return path.read_text(self.encoding, self.errors)
            else:
                return path.read_bytes()
        except FileNotFoundError:
            raise KeyError(key) from None

    def __setitem__(self, key, value):
        path = self.dirpath / key
        if self.text_mode:
            path.write_text(value, self.encoding, self.errors)
        else:
            path.write_bytes(value)


    def __delitem__(self, key):
        try:
            path = self.dirpath / key
            path.unlink()
        except FileNotFoundError:
            raise KeyError(key) from None

    def __contains__(self, key):
        path = self.dirpath / key
        return path.is_file()

    def __iter__(self):
        return ( f.name for f in self.dirpath.iterdir() if f.is_file() )

    def __len__(self):
        return sum(1 for _ in self)

class FileDictExt(KeyMapDict, FileDict):
    """File based dictionary

       A dictionary-like object based on the file system rather than
       in-memory hash tables. It is persistent and sharable between
       proceses.

       Handle only files with a given extention"""

    def __init__(self,*arg, ext=None, **karg):
        super().__init__(*arg,**karg)
        self.ext = ext

    @property
    def ext(self):
        """File extension of the files"""
        return self._ext

    @ext.setter
    def ext(self,value:str):
        if value is None or isinstance(value,str):
            if value and not value.startswith("."):
                raise ValueError("the extension must began with a '.' ex: '.txt' ")
            self._ext = value or ""
        else:
            raise TypeError("value for the ext must be a str")

    @property
    def config(self):
        config = super().config
        config["ext"] = self.ext
        return config

    def keymap(self, key) -> str:
        return f"{key}{self.ext}"

    def __iter__(self):
        it = super().__iter__()
        if self.ext:
            ext = self.ext
            n   = -len(ext)
            return ( key[:n] for key in it if key.endswith(ext) )
        return it

class SerializerDict(ValueMapDict, _ABC.ConfigClass):
    """Cooperative class to store its values in a serialize way"""

    def __init__(self, *arg, serializer=None, read_config:dict=None, write_config:dict=None, **karg):
        super().__init__(*arg, **karg)
        self.serializer = serializer
        self.read_config = read_config
        self.write_config = write_config

    @property
    def serializer(self):
        """modulo/class to serialize data,
           it must have .dumps and .loads functions/methods."""
        return self._serializer

    @serializer.setter
    def serializer(self, value):
        if value:
            if not hasattr(value,"dumps"):
                raise AttributeError("method dumps isn't present")
            if not hasattr(value,"loads"):
                raise AttributeError("method loads isn't present")
        self._serializer = value

    @property
    def read_config(self) -> dict:
        """keyword arguments for self.serializer.loads"""
        return self._read_config

    @read_config.setter
    def read_config(self, value):
        if value and not isinstance(value,abc.Mapping):
            raise TypeError
        self._read_config = dict(value or () )

    @property
    def write_config(self) -> dict:
        """keyword arguments for self.serializer.dumps"""
        return self._write_config

    @write_config.setter
    def write_config(self, value):
        if value and not isinstance(value,abc.Mapping):
            raise TypeError
        self._write_config = dict(value or () )


    @property
    def config(self) -> dict:
        config = super().config
        config["serializer"] = self.serializer
        config["read_config"] = self.read_config
        config["write_config"] = self.write_config
        return config

    def valuemapread(self, value):
        if (s:=self.serializer):
            return s.loads(value, **self.read_config)
        return super().valuemapread(value)

    def valuemapwrite(self, value):
        if (s:=self.serializer):
            return s.dumps(value, **self.write_config)
        return super().valuemapwrite(value)

class FileSerializerDict(_ABC.PropertyConfig, SerializerDict, FileDictExt):
    """A file dict that support serialesing data"""
    pass

class FileDictJson(FileSerializerDict):
    """File based dictionary

       A dictionary-like object based on the file system rather than
       in-memory hash tables and store the values as a json.

       It is persistent and sharable between
       proceses
       """
    ext = ".json"
    serializer = _json
    write_config = dict(sort_keys=True, indent=4)

class FileDictPickle(FileSerializerDict):
    """File based dictionary

       A dictionary-like object based on the file system rather than
       in-memory hash tables and store the values as a pickle.

       It is persistent and sharable between
       proceses
       """
    ext = ".pickle"
    serializer = _pickle
    text_mode = False



