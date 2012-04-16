#!/usr/bin/env python

from collections import defaultdict
import sys

def readAlignmentString(str):
  """
  Read a string of f-e links an return a dictionary
  of link tuples (f,e)
  """
  d = { }
  for link in str.split():
    try:
      f, e = link.split('-')
      d[(int(f), int(e))] = True
    except:
      sys.stderr.write("Couldn't process link '%s'\n" %(link))
      sys.stderr.write("Alignment: %s\n" %(str))
      sys.exit(1)
  return d

class Alignment(object):
  def __init__(self, str):
    self.score = 0
    self.scoreVector = []
    self.links_dict = { }
    # Index links by column, or e index
    self.eLinks = defaultdict(list)
    # Index links also by row, or f index
    self.numLinksInSpan = { }
    self.linksInSpan = { }
    self.read(str)

  def read(self, links_str, delim = '-'):
    """
    Reads and records a string encoded sequence of links, f-e f-e f-e ...
    """
    for linkstr in links_str.strip().split():
      f, e = map(int, linkstr.split(delim))
      link = (f,e)
      self.eLinks[e].append(link)
      self.links_dict[link] = True

  def getLinksByEIndex(self, span):
    """
    Return a list of links (f,e) s.t. span[0] <= e <= span[1]
    """
    links = self.linksInSpan.get(span, None)
    if links is None:
      links = []
      for e in range(span[0], span[1]+1):
        links += self.eLinks[e]
      self.linksInSpan[span] = links
    return links
