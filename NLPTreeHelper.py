#!/usr/env python

from NLPTree import NLPTree
from TerminalNode import TerminalNode
import mpi
import re
import sys
import weakref

################################################################################
# stringToTree(self, str):
# Reed and create a Python NLPTree out of a tree encoded as a PTB-style string
################################################################################


def stringToTreeNoHeadInfo(str):
    ''' read PTB-style tree string and become the tree that the string encodes '''
    # Reset current class members
    rootNode = NLPTree()
    rootNode.data = None
    rootNode.parent = None
    rootNode.children = []
    rootNode.terminals = []
    #rootNode.root = rootNode
    currentRoot = rootNode
    eIndex = -1

    # Blank/Empty Tree
    if str.rstrip() == '0':
        return currentRoot;

    beginPattern = r'^\([^[\)]+)$'
    endPattern = r'^([^\)])*\)$'
    i = 0

    for token in str.split():
        if token[-1] == ')' and len(token) > 1:
            # Close subtree and add terminal node
            # Increment e-index counter
            eIndex += 1
            tokenName = token[0:-1]

            # Create new child and add to tree
            newChild = TerminalNode(tokenName.lower())
            newChild.eIndex = eIndex
            newChild.parent = currentRoot
            newChild.parent.eIndex = eIndex

            # Add new child to the tree
            currentRoot.addChild(newChild)
            # Close subtree and back up to parent
            currentRoot = currentRoot.parent

        elif token[-1] == ')' and len(token) == 1:
            # Close subtree
            # Back up root pointer one level
            if currentRoot.parent is not None:
                currentRoot = currentRoot.parent

        else: # token must begin with '('
            # Begin new subtree
            tokenName = token[1:]
            newChild = NLPTree(tokenName)
            newChild.parent = currentRoot
            if rootNode.data is None:
                rootNode.data = tokenName
            else:
                currentRoot.addChild(newChild)
                currentRoot = newChild

    return currentRoot
################################################################################
# stringToTree(self, str):
# Reed and create a Python NLPTree out of a tree encoded as a PTB-style string
################################################################################


def stringToTree(str):
    ''' read PTB-style tree string and become the tree that the string encodes '''
    # Reset current class members
    rootNode = NLPTree()
    rootNode.data = None
    rootNode.parent = None
    rootNode.children = []
    rootNode.terminals = []
    #rootNode.root = rootNode
    currentRoot = rootNode
    eIndex = -1


    # Blank/Empty Tree
    if str.rstrip() == '0':
        return currentRoot;

    # Remove scores from Radu-tree format
    raduSuffix = r'(~\d+~\d+)\s[-]?[\d\.]+'
    str = re.sub(raduSuffix, r'\g<1>', str)

    try:
        for token in str.split():
            if token[-1] == ')' and len(token) > 1:
                # Close subtree and add terminal node
                # Increment e-index counter
                eIndex += 1
                tokenName = token[0:-1]

                # Create new child and add to tree
                newChild = TerminalNode(tokenName.lower())
                newChild.eIndex = eIndex
                newChild.parent = currentRoot
                newChild.parent.eIndex = eIndex

                # Add new child to the tree
                currentRoot.addChild(newChild)
                # Close subtree and back up to parent
                currentRoot = currentRoot.parent

            elif token[-1] == ')' and len(token) == 1:
                # Close subtree
                # Back up root pointer one level
                if currentRoot.parent is not None:
                    currentRoot = currentRoot.parent

            else: # token must begin with '('
                # Begin new subtree
                tokenName = token[1:]
                # Extract head info if available
                headInfo = tokenName.split('~')
                numEligibleHeads = 0
                headItem = -1
                if len(headInfo) == 3:
                    tokenName = headInfo[0]
                    numEligibleHeads = int(headInfo[1])
                    headItem = int(headInfo[2])

                newChild = NLPTree(tokenName)
                newChild.numEligibleHeads = numEligibleHeads
                newChild.headItem = headItem
                newChild.parent = currentRoot
                if rootNode.data is None:
                    rootNode.data = tokenName
                    rootNode.headItem = headItem
                    rootNode.numEligibleHeads = numEligibleHeads
                else:
                    currentRoot.addChild(newChild)
                    currentRoot = newChild
    except:
        # Upon any error processing the string, assume it is a malformed tree and return an empty tree.
        sys.stderr.write("Malformed tree string: %s\n" % (str))
        rootNode = NLPTree()
        rootNode.data = None
        rootNode.parent = None
        rootNode.children = []
        rootNode.terminals = []
        #rootNode.root = rootNode
        return rootNode

    tree = addSpans(currentRoot)
    return tree

def stringToTree_weakRef(str):
    ''' read PTB-style tree string and become the tree that the string encodes '''
    # Reset current class members
    rootNode = NLPTree()
    rootNode.data = None
    rootNode.parent = None
    rootNode.children = []
    rootNode.terminals = []
    #rootNode.root = rootNode
    currentRoot = rootNode
    eIndex = -1


    # Blank/Empty Tree
    if str.rstrip() == '0':
        return currentRoot;

    # Remove scores from Radu-tree format
    raduSuffix = r'(~\d+~\d+)\s[-]?[\d\.]+'
    str = re.sub(raduSuffix, r'\g<1>', str)
    try:
        for token in str.split():
            if token[-1] == ')' and len(token) > 1:
                # Close subtree and add terminal node
                # Increment e-index counter
                eIndex += 1
                tokenName = token[0:-1]

                # Create new child and add to tree
                newChild = TerminalNode(tokenName.lower())
                newChild.eIndex = eIndex
                newChild.parent = weakref.ref(currentRoot)
                newChild.parent().eIndex = eIndex

                # Add new child to the tree
                currentRoot.addChild(newChild)
                # Close subtree and back up to parent
                currentRoot = currentRoot.parent()

            elif token[-1] == ')' and len(token) == 1:
                # Close subtree
                # Back up root pointer one level
                if currentRoot.parent is not None:
                    currentRoot = currentRoot.parent()

            else: # token must begin with '('
                # Begin new subtree
                tokenName = token[1:]
                # Extract head info if available
                headInfo = tokenName.split('~')
                numEligibleHeads = 0
                headItem = -1
                if len(headInfo) == 3:
                    tokenName = headInfo[0]
                    numEligibleHeads = int(headInfo[1])
                    headItem = int(headInfo[2])

                newChild = NLPTree(tokenName)
                newChild.numEligibleHeads = numEligibleHeads
                newChild.headItem = headItem
                newChild.parent = weakref.ref(currentRoot)
                if rootNode.data is None:
                    rootNode.data = tokenName
                    rootNode.headItem = headItem
                    rootNode.numEligibleHeads = numEligibleHeads
                else:
                    currentRoot.addChild(newChild)
                    currentRoot = newChild
    except:
        # Upon any error processing the string, assume it is a malformed tree and return an empty tree.
        sys.stderr.write("Malformed tree string: %s\n" % (str))
        rootNode = NLPTree()
        rootNode.data = None
        rootNode.parent = None
        rootNode.children = []
        rootNode.terminals = []
        #rootNode.root = rootNode
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

################################################################################
# Helper function containsSpan(currentNode, span)
# Does span of node currentNode wholly contain span?
################################################################################
def containsSpan(currentNode, fspan):
    span = currentNode.get_span()
    return span[0] <= fspan[0] and span[1] >= fspan[1]

if __name__ == "__main__":
    for treestr in sys.stdin:
      tree = stringToTree(treestr)
      print tree

      #    treestr = "(TOP~1~1 0 (S~2~2 0 (NP~1~1 (DT the) (NN man) ) (VP~1~1 (VBD ate) ) ) )"
