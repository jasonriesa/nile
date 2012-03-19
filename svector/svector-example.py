import svector
import cPickle as pickle

# The svector module has only one class, Vector.
# A Vector is essentially a mapping from strings to floats.
# They can be created in various ways:

v1 = svector.Vector()                   # empty
v2 = svector.Vector("foo=1 bar=2")      # from str; most convenient but slowest
v3 = svector.Vector({"foo":3, "bar":4}) # from dict
v4 = svector.Vector(v2)                 # copy from another Vector

# convert a Vector to a string:
print str(v2)
print v2

# pickling and unpickling:
s = pickle.dumps(v2)
print pickle.loads(s)

# Vectors behave like Python containers:
print v2['foo']   # get value (1.)
print len(v2)     # number of key/value pairs (2)
print v1['foo']   # default value (0.)
print len(v1)     # = 0. Note that 'foo' wasn't inserted.
v1['foo'] = 0.    # set value
print len(v1)     # = 1. Note the difference between a zero value and no value.
print 'bar' in v1 # = False
del v1['foo']
print len(v1)     # = 0
for x in v2:
    print x
for x in v2.iterkeys():
    print x
for x,y in v2.iteritems():
    print x,y
for y in v2.itervalues():
    print y

# Algebraic operations

# Nondestructive
print v2+v3       # vector addition
print v2-v3       # vector subtraction
print 2.*v2       # scalar product
print v2*2.       # scalar product
print -v2         # same as -1.*v2
print v2/2.       # scalar division
print v2.dot(v3)  # dot product
print v2*v3       # Hadamard (elementwise) product
print v2/v3       # elementwise division
print v2.norm()   # Euclidean norm

# Destructive
v2 += v3
print v2
v2 -= v3
print v2
v2 *= 2.
print v2
v2 /= 2.
print v2
v2 *= v3
print v2
v2 /= v3
print v2
