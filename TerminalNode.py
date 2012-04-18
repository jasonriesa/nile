#!/usr/bin/env python
# riesa@isi.edu (Jason Riesa)
# Python tree class for reading and processing PTB-style trees

import sys
import weakref
from Tree import Tree
from NLPTree import NLPTree

class TerminalNode(NLPTree):
  def setup(self, data, children=None):
    self.partialAlignments = []
    self.partialAlignments_hope = []
    self.partialAlignments_fear = []
    self.data = data
    self.parent = None
    self.children = [ ]
    for ci, child in enumerate(self.children):
      child.parent = weakref.ref(self)
      child.order = ci
    self.terminals = [ ]
    self.eIndex = -1
    self.oracle = None

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
    Return terminal with index i
    Store only weak references to terminals
    """
    return self.terminals[i]()

  def getTerminals(self):
    """
    Iterator over terminals.
    """
    for t in self.terminals:
      yield t()

  def dfs_write_ptb(self):
    """
    Print the tree in preorder
    """

    if len(self.children) > 0:
      separator_begin = "("
      separator_end = ")"
    else:
      separator_begin = ""
      separator_end = ""

    sys.stdout.write("%s%s" % (separator_begin, self.data))
    for child in self.children:
      sys.stdout.write(" ")
      child.dfs_write_ptb()
    sys.stdout.write("%s" % (separator_end))

  def span_start(self):
    while(self.children):
      return self.children[0].span_start()
    if(len(self.children) == 0):
      return self.eIndex

  def span_end(self):
    while(self.children):
      return self.children[-1].span_end()
    if(len(self.children) == 0):
      return self.eIndex
