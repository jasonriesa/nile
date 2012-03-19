# PartialGridAlignment.py
from collections import defaultdict
import svector

class PartialGridAlignment(object):
    def __hash__(self):
        return id(self)

    def __cmp__(self, x):
        # Implemented s.t. min value is best
        if self.score < x.score:      return -1
        elif self.score == x.score:   return 0
        else:                         return 1

    def __str__(self):
        return str(self.score)+" "+str(self.links)

    def __init__(self, flen = 0, elen = 0):
        ''' class constructor '''
        self.setup(flen, elen)

    def setup(self, flen, elen):
        # 7-17-2010 keep track of separate local and nonlocal feature vectors.
        # I should have called these featureVector instead of scoreVector.
        self.links = [ ]
        self.score = 0
        self.fscore = 0
        self.hope = 0
        self.fear = 0
        # local feature vector
        self.scoreVector = svector.Vector()
        self.scoreVector_nonlocal = svector.Vector()
        self.position = None
        self.boundingBox = None

    def getStringEncoding(self):
        links_str = []
        for link in self.links:
            links_str.append(str(link[0])+"-"+str(link[1]))
        return " ".join(links_str)

    def linksToString(links):
        links_str = []
        for link in links:
            links_str.append(str(link[0])+"-"+str(link[1]))
        return " ".join(links_str)

    def clear(self):
        self.links = []
        self.score = 0
        self.fscore = 0
        self.hope = 0
        self.fear = 0
        self.scoreVector = svector.Vector()
        self.position = None
        self.boundingBox = None
