#!/usr/bin/env python
# hminghkm.py
# riesa@isi.edu (Jason Riesa)
#
# Minimal GHKM-rule extractor from David Chiang
# Modified to use NLPTrees, tree and alignment fragments,
# weakRef objects, and constrained extraction.

"""Command-line usage:

stdin:

COMINGFROM FRANCE AND RUSSIA DE ASTRO NAUTS \t (NP (NP (NNS astronauts)) (VP (VBG coming) (PP (IN from) (NP (NP (NNP france) (CC and) (NNP russia)))))) \t 0-1 0-2 1-3 2-4 3-5 5-0 6-0

stdout:

(NNS NAUTS ASTRO) <- (NNS astronauts)
(NP NNS:x0) <- (NP NNS:x0)
(NNP FRANCE) <- (NNP france)
(CC AND) <- (CC and)
(NNP RUSSIA) <- (NNP russia)
(NP CC:x1 NNP:x2 NNP:x0) <- (NP NNP:x0 CC:x1 NNP:x2)
(NP NP:x0) <- (NP NP:x0)
(VP NP:x0 COMINGFROM) <- (VP (VBG coming) (PP (IN from) NP:x0))
(NP DE NP:x1 VP:x0) <- (NP NP:x1 VP:x0)
"""

import itertools
import sys

import NLPTree
import NLPTreeHelper
import Tree

class Rule(object):
  def __init__(self, f, e):
    """
    f is a one-level tree representing the French side of the rule
    e is a tree representing the English side of the rule
    each node is labeled with either a str or a Variable
    """
    self.f = f
    self.e = e

  def __str__(self):
    return "%s -> %s" % (self.e, " ".join([str(item) for item in self.f.children]))
  def __eq__(self, other):
    return str(self) == str(other)
  def __hash__(self):
    return hash(str(self))

class Variable(object):
  def __init__(self, data, index=None):
    """
    label is the label (syntactic category) of the variable,
    index is a number by which this variable will be paired
    with another variable on the other side of the rule
    """
    self.data = data
    self.index = index

  def __str__(self):
    if self.index is not None:
      return "%s:x%d" % (self.data, self.index)
    else:
      return self.data

  def __eq__(self, other):
    return (self.data, self.index) == (other.data, other.index)
  def __hash__(self):
    return hash((self.data, self.index))

def mark_phrases(fwords, etree, align, offset = 0, hierarchical = False):
  fn = len(fwords)
  en = etree.j - offset

  # the first French word aligned to each English word
  emin = [fn] * en
  # the last French word aligned to each English word, plus one
  emax = [-1] * en

  # the number of English words aligned to each French word
  fcount = [0] * fn
  # similarly for the other direction
  ecount = [0] * en

  for (fi,ei) in align:
    emin[ei] = min(emin[ei],fi)
    emax[ei] = max(emax[ei],fi+1)
    fcount[fi] += 1
    ecount[ei] += 1

  # fcumul[fi] is the number of alignment points in fwords[:fi]
  fcumul = [0]
  s = 0
  for c in fcount:
    s += c
    fcumul.append(s)

  for node in etree.bottomup():
    if len(node.children) == 0:
      node.ecount = ecount[node.i-offset]
      node.emin = emin[node.i-offset]
      node.emax = emax[node.i-offset]
    else:
      node.ecount = 0
      node.emin = fn
      node.emax = -1
      for child in node.children:
        node.emin = min(node.emin, child.emin)
        node.emax = max(node.emax, child.emax)
        node.ecount += child.ecount

    # We know how many alignment points the English node has,
    # and the number of alignment points that the corresponding
    # French span has. If those numbers are equal, then the
    # two are exclusively aligned to each other and we have a phrase.

    node.phrase = node.ecount > 0 and node.ecount == fcumul[node.emax]-fcumul[node.emin]

def _detach_phrases(node, accum, etree, hierarchical):
  copy = NLPTree.NLPTree(node.data, [_detach_phrases(child, accum, etree, hierarchical) for child in node.children])
  copy.emin, copy.emax = node.emin, node.emax
  if node.phrase and len(node.children) > 0:
    # We only care about the current node
    if node == etree or (not hierarchical):
      accum.append(copy)
    copy = NLPTree.NLPTree(Variable(node.data))
    copy.emin, copy.emax = node.emin, node.emax

  return copy

def detach_phrases(node, etree, hierarchical):
  accum = []
  _detach_phrases(node, accum, etree, hierarchical)
  return accum

def make_rule(fwords, etree, align):
  fspans = []

  # build the erhs while keeping track of the French spans of the vars
  for enode in etree.frontier():
    if isinstance(enode.data, Variable):
      fspans.append((enode.emin, enode.emax, enode))

  fspans.sort()

  # build the frhs with links to erhs, and build ants
  r = Rule(NLPTree.NLPTree(etree.data), etree)

  prev_fj = etree.emin
  for (vi,(fi,fj,enode)) in enumerate(fspans):
    assert fj > fi
    for fk in xrange(prev_fj,fi):
      fleaf = NLPTree.NLPTree(fwords[fk])
      r.f.insert_child(-1, fleaf)

    var = Variable(enode.data, vi)
    fleaf = NLPTree.NLPTree(var)
    enode.data = var
    r.f.insert_child(-1, fleaf)

    prev_fj = fj

  for fk in xrange(prev_fj,etree.emax):
    fleaf = NLPTree.NLPTree(fwords[fk])
    r.f.insert_child(-1, fleaf)

  yield r

def extract(fwords, etree, align, offset = 0, hierarchical = False):
  """
  fwords is a list of French words
  etree is an English tree
  align is a list of pairs of (french, english) positions

  returns: an iterator over extracted Rules
  """
  # find the frontier set
  mark_phrases(fwords, etree, align, offset)
  # push outermost unaligned French words into top rule
  etree.emin, etree.emax = 0, len(fwords)
  for subtree in detach_phrases(etree, etree, hierarchical):
    for r in make_rule(fwords, subtree, align):
      yield r

def extractRuleDict(fwords, etree, align, offset = 0, hierarchical = False):
    """
    fwords is a list of French words
    etree is an English tree
    align is a list of pairs of (french, english) positions

    returns: an dictionary of extracted Rules
    """
    rules = { }
    # find the frontier set
    mark_phrases(fwords, etree, align, offset)
    # push outermost unaligned French words into top rule
    etree.emin, etree.emax = 0, len(fwords)
    for subtree in detach_phrases(etree, etree, hierarchical):
      for r in make_rule(fwords, subtree, align):
        rules[r] = True
    return rules

if __name__ == "__main__":
  class SkipSentence(Exception):
    pass

  progress = 0
  skipped = 0

  for line in sys.stdin:
    try:
      (fline, eline, aline) = line.split("\t")
      fwords = fline.split()

      etree = NLPTreeHelper.stringToTree_weakRef(eline)
      if etree is None:
        raise SkipSentence
      if etree.data in ["", "TOP", "ROOT"]:
        if len(etree.children) == 1:
          etree = etree.children[0]
        else:
          sys.stderr.write("warning, line %d: top node has multiple children\n" % (progress+1))
          raise SkipSentence

      fn = len(fwords)
      en = etree.j

      align = []
      for pair in aline.split():
        i,j = pair.split("-",1)
        i = int(i)
        j = int(j)
        if i >= fn or j >= en:
          sys.stderr.write("warning, line %d: alignment point out of bounds\n" % (progress+1))
          raise SkipSentence
        align.append((i,j))

      if len(align) == 0:
        sys.stderr.write("warning, line %d: no alignments\n" % (progress+1))
        raise SkipSentence

      for r in extract(fwords, etree, align):
        print r
    except SkipSentence:
      skipped += 1

    progress += 1

