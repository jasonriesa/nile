cdef class Vector

cimport numberizer

cdef extern from "string":
    ctypedef struct c_string "std::string":
        void assign(char *)
        char *c_str ()

cdef extern from "cmath":
    double sqrt "std::sqrt" (double)

cdef extern from "svector.hpp":
    ctypedef struct c_svector "svector<int,double>":
        void swap(c_svector)
        void erase(int)
        int size()
        void iadd "operator+=" (c_svector)
        void isub "operator-=" (c_svector)
        void imul "operator*=" (double)
        void idiv "operator/=" (double)
        c_svector mul "operator*" (c_svector)
        c_svector div "operator/" (c_svector)
        double dot (c_svector)
        double normsquared ()

    c_svector *new_svector "new svector<int,double>" ()
    c_svector *new_svector2 "new svector<int,double>" (int, double)
    c_svector *copy_svector "new svector<int,double>" (c_svector)
    c_svector *svector_from_str "svector_from_str<int,double>" (c_string)
    void del_svector "delete" (c_svector *)

cdef extern from "cy_svector.hpp":
    void svector_setitem (c_svector, int, double)
    double svector_getitem (c_svector, int)
    bint svector_contains (c_svector, int)
    c_string svector_str(c_svector) except +
    numberizer.c_numberizer feature_numberizer

    ctypedef struct c_svector_iterator "svector_iterator<int,double>":
        bint has_next()
        c_string key() except +
        double value()
        void next()

    c_svector_iterator* new_svector_iterator "new svector_iterator<int,double>" (c_svector)
    void del_svector_iterator "delete" (c_svector_iterator *)

cdef int word_to_index(char *s):
    cdef c_string cs
    cs.assign(s)
    return feature_numberizer.word_to_index(cs)

cdef class KeyIterator:
    cdef c_svector_iterator *thisptr
    cdef Vector v
    
    def __cinit__(self, Vector v):
        self.thisptr = new_svector_iterator(v.thisptr[0])
        self.v = v # keep reference
    def __dealloc__(self):
        del_svector_iterator(self.thisptr)
    def __iter__(self):
        return self
    def __next__(self):
        cdef c_string key
        if self.thisptr.has_next():
            key = self.thisptr.key()
            self.thisptr.next()
            return key.c_str()
        else:
            raise StopIteration()

cdef class ValueIterator:
    cdef c_svector_iterator *thisptr
    cdef Vector v

    def __cinit__(self, Vector v):
        self.thisptr = new_svector_iterator(v.thisptr[0])
        self.v = v # keep reference
    def __dealloc__(self):
        del_svector_iterator(self.thisptr)
    def __iter__(self):
        return self
    def __next__(self):
        cdef double value
        if self.thisptr.has_next():
            value = self.thisptr.value()
            self.thisptr.next()
            return value
        else:
            raise StopIteration()

cdef class KeyValueIterator:
    cdef c_svector_iterator *thisptr
    cdef Vector v

    def __cinit__(self, Vector v):
        self.thisptr = new_svector_iterator(v.thisptr[0])
        self.v = v # keep reference
    def __dealloc__(self):
        del_svector_iterator(self.thisptr)
    def __iter__(self):
        return self
    def __next__(self):
        cdef c_string key
        cdef double value
        if self.thisptr.has_next():
            key = self.thisptr.key()
            value = self.thisptr.value()
            self.thisptr.next()
            return (key.c_str(), value)
        else:
            raise StopIteration()

cdef class Vector:
    cdef c_svector *thisptr

    def __init__(self, *args):
        cdef Vector vx
        cdef c_string s

        del_svector(self.thisptr) # in case __init__ is called twice
        
        if len(args) == 0:
            self.thisptr = new_svector()
        elif len(args) == 1:
            x = args[0]
            if isinstance(x, str):
                s.assign(x)
                self.thisptr = svector_from_str(s)
            elif isinstance(x, Vector):
                vx = x
                self.thisptr = copy_svector(vx.thisptr[0])
            elif isinstance(x, dict):
                self.thisptr = new_svector()
                for f,v in x.iteritems():
                    svector_setitem(self.thisptr[0], word_to_index(f), v)
            else:
                raise TypeError()
        elif len(args) == 2:
            f, x = args
            self.thisptr = new_svector2(word_to_index(f), x)
        else:
            raise TypeError()

    def __dealloc__(self):
        del_svector(self.thisptr)

    def __copy__(self):
        return Vector(self)

    def __str__(self):
        cdef c_string cs = svector_str(self.thisptr[0])
        return cs.c_str()

    def __reduce__(self):
        cdef c_string cs = svector_str(self.thisptr[0])
        return (Vector, (cs.c_str(),))

    # container interface

    def __len__(self):
        return self.thisptr.size()

    def __setitem__(self, char *f, double x):
        svector_setitem(self.thisptr[0], word_to_index(f), x)

    def __getitem__(self, char *f):
        return svector_getitem(self.thisptr[0], word_to_index(f))

    def __delitem__(self, char *f):
        self.thisptr.erase(word_to_index(f))

    def __iter__(self):
        return KeyIterator(self)

    def iterkeys(self):
        return KeyIterator(self)

    def itervalues(self):
        return ValueIterator(self)

    def iteritems(self):
        return KeyValueIterator(self)

    def __contains__(self, char *f):
        return svector_contains(self.thisptr[0], word_to_index(f))
        
    # algebraic operations

    def __iadd__(self, Vector other):
        self.thisptr.iadd(other.thisptr[0])
        return self
    
    def __isub__(self, Vector other):
        self.thisptr.isub(other.thisptr[0])
        return self

    def __imul__(self, other):
        if isinstance(other, Vector):
            return self.__mul__(other)
        else:
            self.thisptr.imul(float(other))
            return self
    
    def __idiv__(self, other):
        if isinstance(other, Vector):
            return self.__div__(other)
        else:
            self.thisptr.idiv(float(other))
            return self
    
    def __add__(Vector self, Vector other):
        cdef Vector z = Vector(self)
        z.thisptr.iadd(other.thisptr[0])
        return z
    
    def __sub__(Vector self, Vector other):
        cdef Vector z = Vector(self)
        z.thisptr.isub(other.thisptr[0])
        return z
    
    def __mul__(self, other):
        cdef Vector x
        cdef Vector y
        cdef Vector z
        if isinstance(self, Vector) and isinstance(other, Vector): # elementwise product
            x = self
            y = other
            z = Vector()
            del_svector(z.thisptr)
            z.thisptr = copy_svector(x.thisptr.mul(y.thisptr[0]))
            #x.thisptr.mul(y.thisptr[0]).swap(z.thisptr[0])

        elif isinstance(self, Vector): # scalar product
            z = Vector(self)
            z.thisptr.imul(float(other))
        elif isinstance(other, Vector): # reverse scalar product
            z = Vector(other)
            z.thisptr.imul(float(self))
        return z
    
    def __div__(self, other):
        cdef Vector x
        cdef Vector y
        cdef Vector z
        if isinstance(other, Vector): # elementwise divide
            x = self
            y = other
            z = Vector()
            del_svector(z.thisptr)
            z.thisptr = copy_svector(x.thisptr.div(y.thisptr[0]))
            #x.thisptr.mul(y.thisptr[0]).swap(z.thisptr[0])
        else:
            z = Vector(self)
            z.thisptr.idiv(float(other))
        return z

    def __neg__(Vector self):
        cdef Vector z = Vector(self)
        z.thisptr.imul(-1)
        return z

    def __pos__(Vector self):
        cdef Vector z = Vector(self)
        return z

    def dot(Vector self, Vector other):
        return self.thisptr.dot(other.thisptr[0])
    
    def norm(Vector self):
        return sqrt(self.thisptr.normsquared())

