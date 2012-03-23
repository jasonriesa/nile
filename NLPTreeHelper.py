#!/usr/env python

import re
import sys
import weakref

from NLPTree import NLPTree
from TerminalNode import TerminalNode

def stringToTree_weakRef(str):
    """
    Read PTB-style tree string and return the tree that the string encodes.
    """
    # Reset current class members
    rootNode = NLPTree()
    rootNode.data = None
    rootNode.parent = None
    rootNode.children = []
    rootNode.terminals = []
    currentRoot = rootNode
    eIndex = -1

    # Blank/Empty Tree
    if str.rstrip() == '0':
        return currentRoot;
    try:
      for token in str.split():
        if token[-1] == ')' and len(token) > 1:
          # Close subtree and add terminal node
          # Increment e-index counter
          eIndex += 1
          # How many levels are we closing?
          closingMarkerLocation = token.find(')')
          levelsToClose = len(token) - closingMarkerLocation
          # Get token name
          tokenName = token[0:closingMarkerLocation]

          # Create new child and add to tree
          newChild = TerminalNode(tokenName.lower())
          newChild.eIndex = eIndex
          newChild.parent = weakref.ref(currentRoot)
          newChild.parent().eIndex = eIndex

          # Add new child to the tree
          currentRoot.addChild(newChild)
          # Close subtree and back up to parent
          while levelsToClose > 0 and currentRoot.parent is not None:
            currentRoot = currentRoot.parent()
            levelsToClose -= 1
        else:
          # token must begin with '('
          # Begin new subtree
          tokenName = token[1:]
          newChild = NLPTree(tokenName)
          newChild.parent = weakref.ref(currentRoot)
          if rootNode.data is None:
            rootNode.data = tokenName
          else:
            currentRoot.addChild(newChild)
            currentRoot = newChild
    except Exception as e:
      # Upon any error processing the string,
      # assume for now it is due to a malformed tree string
      # and return an empty tree.
      sys.stderr.write("Potentially malformed tree string: %s\n" % (str))
      print e
      rootNode = NLPTree()
      rootNode.data = None
      rootNode.parent = None
      rootNode.children = []
      rootNode.terminals = []
      return rootNode
    tree = addSpans(currentRoot)
    return tree

def addSpans(tree):
  i = 0
  for node in tree.bottomup():
    if len(node.children) == 0:
      node.i = i
      node.j = i+1
      i += 1
    else:
      node.i = node.children[0].i
      node.j = node.children[-1].j

  return tree

def containsSpan(currentNode, fspan):
  """
  Does span of node currentNode wholly contain span fspan?
  """
  span = currentNode.get_span()
  return span[0] <= fspan[0] and span[1] >= fspan[1]

if __name__ == "__main__":
    treestr = "(TOP (S (NP (DT the) (NN man)) (VP (VBD ate))))"
    tree = stringToTree_weakRef(treestr)
    print tree
