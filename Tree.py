import weakref

class Tree(object):
  def __init__(self, data = None, children=None):
    self.setup(data, children)

  def setup(self, data, children = None):
    self.data = data
    self.parent = None
    self.children = []
    for ci, child in enumerate(self.children):
      child.parent = weakref.ref(self)
      child.order = ci
    self.i = -1
    self.j = -1

  def addChild(self, child):
    """
    Add a child subtree to the tree
    """
    child.parent = weakref.ref(self)
    child.order = len(self.children)
    self.children.append(child)

  def getParent(self):
    return self.parent()

  def isTerminal(self):
    return len(self.children) == 0

  def depth(self,d = 0):
    maxDepth = d
    for child in self.children:
      childDepth = child.depth(d+1)
      if childDepth+1 > maxDepth:
        maxDepth = childDepth
    return maxDepth

  def bottomup(self, returnOnlySelf = False):
    for child in self.children:
      for node in child.bottomup():
        yield node
    yield self

  def detach(self):
    if self.parent():
      self.parent().delete_child(self.order)

  def delete_child(self, i):
    self.children[i].parent = None
    self.children[i].order = 0
    self.children[i:i+1] = []
    for j in range(i,len(self.children)):
      self.children[j].order = j

  def insert_child(self, i, child):
    child.parent = weakref.ref(self)
    if i != -1:
      self.children[i:i] = [child]
    else:
      self.children[len(self.children):]=[child]
    for j in range(i,len(self.children)):
      self.children[j].order = j

  def frontier(self):
    if len(self.children) != 0:
      l = []
      for child in self.children:
        l.extend(child.frontier())
      return l
    else:
      return [self]

