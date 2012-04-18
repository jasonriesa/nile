#!/usr/bin/env python
# riesa@isi.edu (Jason Riesa)
# Python tree class for reading and processing PTB-style trees

import re
import weakref
from Tree import Tree
from PartialGridAlignment import PartialGridAlignment

class NLPTree(Tree):
  def setup(self, data, children = None):
    self.data = data
    self.span = None
    self.parent = None
    self.span = None
    self.children = [ ]
    if children is not None:
      self.children = children
    for ci, child in enumerate(self.children):
      child.parent = weakref.ref(self)
      child.order = ci

    self.terminals = [ ]
    self.oracle = None
    self.hope = None
    self.fear = None
    self.order = 0
    self.i = -1
    self.j = -1

    # Hold list of PartialGridAlignments associated with this node
    self.partialAlignments = []
    self.partialAlignments_hope = []
    self.partialAlignments_fear = []

  def write(self):
    """
    Print a PTB-style string encoding of the tree
    """
    self.dfs_write_ptb()
    print

  def getPreTerminals(self):
    """
    Return a list of preterminal nodes in the span of this node
    """
    for child in self.children:
      if len(child.children) == 0:
        self.terminals.append(weakref.ref(self))
      else:
        for terminal in child.getPreTerminals():
          self.terminals.append(terminal)

    return self.terminals

  def getTerminal(self, i):
    """
    Return terminal with index i.
    Store only weak references to terminals.
    """
    return self.terminals[i]()

  def getTerminals(self):
    for t in self.terminals:
      yield t()

  def span_start(self):
    if self.span is not None:
      # Dont recompute if we already know the answer
      return self.span[0]

    if len(self.children) > 0:
      return self.children[0].span_start()
    if(len(self.children) == 0):
      return self.eIndex

  def get_span(self):
    if self.span is None:
      start = self.span_start()
      end = self.span_end()
      self.span = (start,end)
    return self.span

  def span_end(self):
    if self.span is not None:
      # Dont recompute if we already know the answer
      return self.span[1]

    if len(self.children) > 0:
      return self.children[-1].span_end()
    if(len(self.children) == 0):
      return self.eIndex

  def isWithinSpan(self,index):
    # Return True if index is within the span of this node
    mySpan = self.get_span()
    return index >= mySpan[0] and index <= mySpan[1]

  def detach(self):
    if self.parent():
      self.parent().delete_child(self.order)

  def delete_child(self, i):
    self.children[i].parent = None
    self.children[i].order = 0
    self.children[i:i+1] = []
    for j in range(i,len(self.children)):
      self.children[j].order = j

  def __str__(self):
    if len(self.children) != 0:
      s = "(" + str(self.data)
      for child in self.children:
        s += " " + child.__str__()
      s += ")"
      return s
    else:
      s = str(self.data)
      s = re.sub("\(", "-LRB-", s)
      s = re.sub("\)", "-RRB-", s)
      return s
