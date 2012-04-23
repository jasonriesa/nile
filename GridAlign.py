#!/usr/bin/env python

#########################################################
# GridAlign.py
# riesa@isi.edu (Jason Riesa)
# Based on work described in:
# @inproceedings{RiesaIrvineMarcu:11,
#   Title = {Feature-Rich Language-Independent Syntax-Based Alignment for Statistical Machine Translation},
#   Author = {Jason Riesa and Ann Irvine and Daniel Marcu},
#   Booktitle = {Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing},
#   Pages = {497--507},
#   Publisher = {Association for Computational Linguistics},
#   Year = {2011}}
#
# @inproceedings{RiesaMarcu:10,
#   Title = {Hierarchical Search for Word Alignment},
#   Author = {Jason Riesa and Daniel Marcu},
#   Booktitle = {Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics (ACL)},
#   Pages = {157--166},
#   Publisher = {Association for Computational Linguistics},
#   Year = {2010}}
#########################################################

import cPickle
import sys
from itertools import izip
from operator import attrgetter
from heapq import heappush, heapify, heappop, heappushpop
from collections import defaultdict

from TerminalNode import TerminalNode
from Alignment import readAlignmentString
from PartialGridAlignment import PartialGridAlignment
from NLPTreeHelper import *
import Fmeasure
import svector
import hminghkm

class Model(object):
  """
  Main class for the Hierarchical Alignment model
  """
  def __init__(self, f = None, e = None, etree = None, ftree = None,
               id = "no-id-given", weights = None, a1 = None, a2 = None,
               inverse = None, DECODING=False,
               LOCAL_FEATURES = None, NONLOCAL_FEATURES = None, FLAGS=None):

    ################################################
    # Constants and Flags
    ################################################
    if FLAGS is None:
      sys.stderr.write("Program flags not given to alignment model.\n")
      sys.exit(1)

    self.FLAGS = FLAGS

    self.LOCAL_FEATURES = LOCAL_FEATURES
    self.NONLOCAL_FEATURES = NONLOCAL_FEATURES
    self.LANG = FLAGS.langpair
    if FLAGS.init_k is not None:
      self.BEAM_SIZE = FLAGS.init_k
    else:
      self.BEAM_SIZE = FLAGS.k
    self.NT_BEAM = FLAGS.k
    self.COMPUTE_HOPE = False
    self.COMPUTE_1BEST = False
    self.COMPUTE_FEAR = False
    self.COMPUTE_ORACLE = False
    self.DO_RESCORE = FLAGS.rescore
    if DECODING:
      self.COMPUTE_1BEST = True
    else:
      if FLAGS.oracle == "gold":
        self.COMPUTE_ORACLE = True
      elif FLAGS.oracle == "hope":
        self.COMPUTE_HOPE = True

      elif FLAGS.oracle is not None:
        # During decoding we don't need to compute oracle
        sys.stderr.write("Unknown value: oracle=%s\n" %(FLAGS.oracle))
        sys.exit(1)

      if FLAGS.hyp == "1best":
        self.COMPUTE_1BEST = True
      elif FLAGS.hyp == "fear":
        self.COMPUTE_FEAR = True
      else:
        sys.stderr.write("Unknown value: hyp=%s\n" %(FLAGS.hyp))
        sys.exit(1)

    # Extra info to pass to feature functions
    self.info = { }

    self.f = f
    self.fstring = " ".join(f)
    self.e = e
    self.lenE = len(e)
    self.lenF = len(f)

    # GIZA++ alignments
    self.a1 = { }		# intersection
    self.a2 = { }		# grow-diag-final
    self.inverse = { }      # ivi-inverse
    if FLAGS.inverse is not None:
      self.inverse = readAlignmentString(inverse)
    if FLAGS.a1 is not None:
      self.a1 = readAlignmentString(a1)
    if FLAGS.a2 is not None:
      self.a2 = readAlignmentString(a2)

    self.modelBest = None
    self.oracle = None
    self.gold = None

    self.id = id
    self.pef = { }
    self.pfe = { }

    self.etree = stringToTree_weakRef(etree)
    self.etree.terminals = self.etree.getPreTerminals()
    if ftree is not None:
      self.ftree = stringToTree_weakRef(ftree)
      self.ftree.terminals = self.ftree.getPreTerminals()
    else:
      self.ftree = None

    # Keep track of all of our feature templates
    self.featureTemplates = [ ]
    self.featureTemplates_nonlocal= [ ]

    ########################################
    # Add weight vector to model
    ########################################
    # Initialize local weights
    if weights is None or len(weights) == 0:
      self.weights = svector.Vector()
    else:
      self.weights = weights

    ########################################
    # Add Feature templates to model
    ########################################
    self.featureTemplateSetup_local(LOCAL_FEATURES)
    self.featureTemplateSetup_nonlocal(NONLOCAL_FEATURES)

    # Data structures for feature function memoization
    self.diagValues = { }
    self.treeDistValues = { }

    # Populate info
    self.info['a1']=self.a1
    self.info['a2']=self.a2
    self.info['inverse']=self.inverse
    self.info['f'] = self.f
    self.info['e'] = self.e
    self.info['etree'] = self.etree
    self.info['ftree'] = self.ftree

  ########################################
  # Initialize feature function list
  ########################################
  def featureTemplateSetup_local(self, localFeatures):
    """
    Incorporate the following "local" features into our model.
    """
    self.featureTemplates.append(localFeatures.ff_identity)
    self.featureTemplates.append(localFeatures.ff_hminghkm)
    self.featureTemplates.append(localFeatures.ff_jumpDistance)
    self.featureTemplates.append(localFeatures.ff_finalPeriodAlignedToNonPeriod)
    self.featureTemplates.append(localFeatures.ff_lexprob_zero)
    self.featureTemplates.append(localFeatures.ff_probEgivenF)
    self.featureTemplates.append(localFeatures.ff_probFgivenE)
    self.featureTemplates.append(localFeatures.ff_distToDiag)
    self.featureTemplates.append(localFeatures.ff_isLinkedToNullWord)
    self.featureTemplates.append(localFeatures.ff_isPuncAndHasMoreThanOneLink)
    self.featureTemplates.append(localFeatures.ff_quote1to1)
    self.featureTemplates.append(localFeatures.ff_unalignedNonfinalPeriod)
    self.featureTemplates.append(localFeatures.ff_nonfinalPeriodLinkedToComma)
    self.featureTemplates.append(localFeatures.ff_nonPeriodLinkedToPeriod)
    self.featureTemplates.append(localFeatures.ff_nonfinalPeriodLinkedToFinalPeriod)
    self.featureTemplates.append(localFeatures.ff_tgtTag_srcTag)
    self.featureTemplates.append(localFeatures.ff_thirdParty)

  ##################################################
  # Inititalize feature function list
  ##################################################
  def featureTemplateSetup_nonlocal(self, nonlocalFeatures):
    """
    Incorporate the following combination-cost features into our model.
    """
    #self.featureTemplates_nonlocal.append(nonlocalFeatures.ff_nonlocal_dummy)
    self.featureTemplates_nonlocal.append(nonlocalFeatures.ff_nonlocal_hminghkm)
    self.featureTemplates_nonlocal.append(nonlocalFeatures.ff_nonlocal_isPuncAndHasMoreThanOneLink)
    self.featureTemplates_nonlocal.append(nonlocalFeatures.ff_nonlocal_sameWordLinks)
    self.featureTemplates_nonlocal.append(nonlocalFeatures.ff_nonlocal_treeDistance1)
    self.featureTemplates_nonlocal.append(nonlocalFeatures.ff_nonlocal_tgtTag_srcTag)
    self.featureTemplates_nonlocal.append(nonlocalFeatures.ff_nonlocal_crossb)

  def align(self):
    """
    Main wrapper for performing alignment.
    """
    ##############################################
    # Do the alignment, traversing tree bottom up.
    ##############################################
    self.bottom_up_visit()
    # *DONE* Now finalize everything; final bookkeeping.

    if self.COMPUTE_1BEST:
      self.modelBest = self.etree.partialAlignments[0]
    if self.COMPUTE_ORACLE:
      self.oracle = self.etree.oracle
    if self.COMPUTE_HOPE:
      self.hope = self.etree.partialAlignments_hope[0]
    if self.COMPUTE_FEAR:
      self.fear = self.etree.partialAlignments_fear[0]

  def bottom_up_visit(self):
    """
    Visit each node in the tree, bottom up, and in level-order.

    ###########################################################
    # bottom_up_visit(self):
    # traverse etree bottom-up, in level order
    # (1) Add terminal nodes to the visit queue
    # (2) As each node is visited, add its parent to the visit
    #     queue if not already on the queue
    # (3) During each visit, perform the proper alignment function
    #     depending on the type of node: 'terminal' or 'non-terminal'
    ###########################################################
    """
    queue = [ ]
    if self.etree.data is None:
      empty = PartialGridAlignment()
      empty.score = None
      self.etree.partialAlignments.append(empty)
      self.etree.oracle = PartialGridAlignment()
      return

    # Add first-level nodes to the queue
    for terminal in self.etree.getTerminals():
      queue.append(terminal)

    # Visit each node in the queue and put parent
    # in queue if not there already
    # Parent is there already if it is the last one in the queue
    while len(queue) > 0:
      currentNode = queue.pop(0)

      # Put parent in the queue if it is not there already
      # We are guaranteed to have visited all of a node's children before we visit that node
      if (currentNode.parent is not None) and (len(queue) == 0 or queue[-1] is not currentNode.parent()):
        if abs(currentNode.parent().depth() - currentNode.depth()) == 1:
          queue.append(currentNode.parent())

      # Visit node here.
      # if currentNode.isTerminal():
      # Is current node a preterminal?
      if len(currentNode.children[0].children) == 0:
        self.terminal_operation(currentNode.eIndex, currentNode)
      else:
        self.nonterminal_operation_cube(currentNode)

  ################################################################################
  # nonterminal_operation_cube(self, currentNode):
  # Perform alignment for visit of nonterminal currentNode
  ################################################################################
  def nonterminal_operation_cube(self, currentNode):
      # To speed up each epoch of training (but not necessarily convergence),
      # generate a single forest with model score as the objective
      # Search through that forest for the oracle hypotheses,
      # e.g. hope (or fear)

      # If there is only one child under currentNode,
      # just copy the contents from child up to currentNode, and move on.
      numChildren = len(currentNode.children)
      if numChildren == 1:
        if self.COMPUTE_1BEST:
          currentNode.partialAlignments = currentNode.children[0].partialAlignments
        if self.COMPUTE_ORACLE:
          currentNode.oracle = currentNode.children[0].oracle
        if self.COMPUTE_HOPE:
          currentNode.partialAlignments_hope = currentNode.children[0].partialAlignments_hope
        if self.COMPUTE_FEAR:
          currentNode.partialAlignments_fear = currentNode.children[0].partialAlignments_fear
        return

      # Compute the span of currentNode
      # span is an ordered pair [i,j] where:
      # i = index of the first eword in span of currentNode
      # j = index of the  last eword in span of currentNode

      span = (currentNode.span_start(), currentNode.span_end())
      currentNode.span = span

      ########################################################################
      # 1-BEST SEARCH
      ########################################################################
      if self.COMPUTE_1BEST:
          # Initialize
          queue = []
          heapify(queue)
          # Before we push, check to see if object's position is in duplicates
          # i.e., we have already visited that position and added the resultant object to the queue
          count = defaultdict(int)
          # Number of components in position vector is the number of children in the current node
          # Position vector uniquely identifies a position in the cube
          # and identifies a unique alignment structure
          position = [0]*len(currentNode.children)

          # Create structure of first object in position [0,0,0,...,0]
          # This path identifies the structure that is the best structure
          # we know of before combination costs (rescoring).
          edges = [ ]
          for c in xrange(numChildren):
            # Object number for current child
            edgeNumber = position[c]
            currentChild = currentNode.children[c]
            edge = currentChild.partialAlignments[edgeNumber]
            edges.append(edge)
          newEdge, boundingBox = self.createEdge(edges, currentNode, span)

          # Where did this new edge come from?
          newEdge.position = list(position)
          # Add new edge to the queue/buffer
          heappush(queue, (newEdge.score*-1, newEdge))

          # Keep filling up my cell until self.BEAM_SIZE has been reached *or*
          # we have exhausted all possible items in the queue
          while(len(queue) > 0 and len(currentNode.partialAlignments) < self.NT_BEAM):
            # Find current best
            (_, currentBestCombinedEdge) = heappop(queue)
            # Add to my cell
            self.addPartialAlignment(currentNode.partialAlignments,
                                     currentBestCombinedEdge,
                                     self.NT_BEAM)
            # Don't create and score more edges when we are already full.
            if len(currentNode.partialAlignments) >= self.NT_BEAM:
              break
            # - Find neighbors
            # - Rescore neighbors
            # - Add neighbors to the queue to be explored
            #   o For every child, there exists a neighbor
            #   o numNeighbors = numChildren
            for componentNumber in xrange(numChildren):
              # Compute neighbor position
              neighborPosition = list(currentBestCombinedEdge.position)
              neighborPosition[componentNumber] += 1
              # Is this neighbor out of range?
              if neighborPosition[componentNumber] >= len(currentNode.children[componentNumber].partialAlignments):
                continue

              # Has this neighbor already been visited?
              #if duplicates.has_key(tuple(neighborPosition)):
              #    continue
              # Lazy eval trick due to Matthias Buechse:
              # Only evaluate after both a node's predecessors have been evaluated.
              # Special case: if any component of neighborPosition is 0, it is on the border.
              # In this case, it only has one predecessor (the one that led us to this position),
              # and can be immediately evaluated.
              if 0 not in neighborPosition and count[tuple(neighborPosition)] < 1:
                count[tuple(neighborPosition)] += 1
                continue

              # Now build the neighbor edge
              neighbor = []
              for cellNumber in xrange(numChildren):
                cell = currentNode.children[cellNumber]
                edgeNumber = neighborPosition[cellNumber]
                edge = cell.partialAlignments[edgeNumber]
                neighbor.append(edge)
              neighborEdge, boundingBox = self.createEdge(neighbor,
                                                          currentNode,
                                                          span)
              neighborEdge.position = neighborPosition
              heappush(queue, (-1*neighborEdge.score, neighborEdge))

          ####################################################################
          # Finalize.
          ####################################################################
          # Sort model score list.
          sortedItems = []
          while(len(currentNode.partialAlignments) > 0):
            sortedItems.insert(0, heappop(currentNode.partialAlignments))
          currentNode.partialAlignments = sortedItems

  	   ## --- end 1best computation --- ##

      if self.COMPUTE_ORACLE:
        # Oracle BEFORE beam is applied.
        # Should just copy oracle up from terminal nodes.
        oracleChildEdges = [c.oracle for c in currentNode.children]
        oracleAlignment, boundingBox = self.createEdge(oracleChildEdges,
                                                       currentNode,
                                                       span)

        # Oracle AFTER beam is applied.
        #oracleCandidates = list(currentNode.partialAlignments)
        #oracleCandidates.sort(key=attrgetter('fscore'),reverse=True)
        #oracleAlignment = oracleCandidates[0]
        currentNode.oracle = oracleAlignment

      ########################################################################
      # HOPE SEARCH
      ########################################################################
      if self.COMPUTE_HOPE:
        # Initialize
        queue_hope = []
        heapify(queue_hope)
        # Before we push, check to see if object's position is in duplicates
        # i.e., we have already visited that position and added the resultant
        # object to the queue
        count_hope = defaultdict(int)
        position = [0]*len(currentNode.children)

        # Create structure of first object in position [0,0,0,...,0]
        # This path implies a resultant structure that is the best structure
        # we know of before combination costs (rescoring).
        edges = [ ]
        for c in xrange(numChildren):
          # Object number for current child
          edgeNumber = position[c]
          currentChild = currentNode.children[c]
          edge = currentChild.partialAlignments_hope[edgeNumber]
          edges.append(edge)
        newEdge, boundingBox = self.createEdge(edges, currentNode, span)
        newEdge.hope = newEdge.score + newEdge.fscore

        # Where did this new edge come from?
        newEdge.position = list(position)
        # Add new edge to the queue/buffer
        heappush(queue_hope, (newEdge.hope*-1, newEdge))

        while(len(queue_hope) > 0 and len(currentNode.partialAlignments_hope) < self.NT_BEAM):
          # Find current best; add to my cell
          (_, currentBestCombinedEdge_hope) = heappop(queue_hope)
          self.addPartialAlignment_hope(currentNode.partialAlignments_hope,
                                        currentBestCombinedEdge_hope,
                                        self.NT_BEAM)
          # Don't create and score more edges when we are already full.
          if len(currentNode.partialAlignments_hope) >= self.NT_BEAM:
              break
          # - Find neighbors
          # - Rescore neighbors
          # - Add neighbors to the queue to be explored
          #   o For every child, there exists a neighbor
          #   o numNeighbors = numChildren
          for componentNumber in xrange(numChildren):
            # Compute neighbor position
            neighborPosition = list(currentBestCombinedEdge_hope.position)
            neighborPosition[componentNumber] += 1
            # Is this neighbor out of range?
            if neighborPosition[componentNumber] >= len(currentNode.children[componentNumber].partialAlignments_hope):
              continue

            # Has this neighbor already been visited?
            #if duplicates_hope.has_key(tuple(neighborPosition)):
            #    continue
            # Lazy eval trick due to Matthias Buechse:
            # Only evaluate after both a node's predecessors have been evaluated.
            # Special case: if any component of neighborPosition is 0, it is on the border.
            # In this case, it only has one predecessor (the one that led us to this position),
            # and can be immediately evaluated.
            if 0 not in neighborPosition and count_hope[tuple(neighborPosition)] < 1:
              count_hope[tuple(neighborPosition)] += 1
              continue

            # Now build the neighbor edge
            neighbor = []
            for cellNumber in xrange(numChildren):
              cell = currentNode.children[cellNumber]
              edgeNumber = neighborPosition[cellNumber]
              edge = cell.partialAlignments_hope[edgeNumber]
              neighbor.append(edge)
            neighborEdge, boundingBox = self.createEdge(neighbor,
                                                        currentNode,
                                                        span)

            neighborEdge.position = neighborPosition
            neighborEdge.hope = neighborEdge.fscore + neighborEdge.score
            heappush(queue_hope, (neighborEdge.hope*-1, neighborEdge))

        sortedItems_hope = []
        while(len(currentNode.partialAlignments_hope) > 0):
          (_, obj) = heappop(currentNode.partialAlignments_hope)
          sortedItems_hope.insert(0, obj)
        currentNode.partialAlignments_hope = sortedItems_hope

      ########################################################################
      # FEAR SEARCH
      ########################################################################
      if self.COMPUTE_FEAR:
        # Initialize
        queue_fear = []
        heapify(queue_fear)
        # Before we push, check to see if object's position is in duplicates
        # i.e., we have already visited that position and added the resultant
        # object to the queue
        count_fear = defaultdict(int)
        position = [0]*len(currentNode.children)

        # Create structure of first object in position [0,0,0,...,0]
        # This path implies a resultant structure that is the best structure
        # we know of before combination costs (rescoring).
        edges = [ ]
        for c in xrange(numChildren):
          # Object number for current child
          edgeNumber = position[c]
          currentChild = currentNode.children[c]
          edge = currentChild.partialAlignments_fear[edgeNumber]
          edges.append(edge)
        newEdge, boundingBox = self.createEdge(edges, currentNode, span)
        newEdge.fear = (1 - newEdge.fscore) + newEdge.score

        # Where did this new edge come from?
        newEdge.position = list(position)
        # Add new edge to the queue/buffer
        heappush(queue_fear, (newEdge.fear*-1, newEdge))

        while(len(queue_fear) > 0 and len(currentNode.partialAlignments_fear) < self.NT_BEAM):
          # Find current best; add to my cell
          (_, currentBestCombinedEdge_fear) = heappop(queue_fear)
          self.addPartialAlignment_fear(currentNode.partialAlignments_fear,
                                        currentBestCombinedEdge_fear,
                                        self.NT_BEAM)
          # Don't create and score more edges when we are already full.
          if len(currentNode.partialAlignments_fear) >= self.NT_BEAM:
            break
          # - Find neighbors
          # - Rescore neighbors
          # - Add neighbors to the queue to be explored
          #   o For every child, there exists a neighbor
          #   o numNeighbors = numChildren
          for componentNumber in xrange(numChildren):
            # Compute neighbor position
            neighborPosition = list(currentBestCombinedEdge_fear.position)
            neighborPosition[componentNumber] += 1
            # Is this neighbor out of range?
            if neighborPosition[componentNumber] >= len(currentNode.children[componentNumber].partialAlignments_fear):
              continue
            # Has this neighbor already been visited?
            #if duplicates_fear.has_key(tuple(neighborPosition)):
            #    continue
            # Lazy eval trick due to Matthias Buechse:
            # Only evaluate after both a node's predecessors have been evaluated.
            # Special case: if any component of neighborPosition is 0, it is on the border.
            # In this case, it only has one predecessor (the one that led us to this position),
            # and can be immediately evaluated.
            if 0 not in neighborPosition and count_fear[tuple(neighborPosition)] < 1:
              count_fear[tuple(neighborPosition)] += 1
              continue

            # Now build the neighbor edge
            neighbor = []
            for cellNumber in xrange(numChildren):
              cell = currentNode.children[cellNumber]
              edgeNumber = neighborPosition[cellNumber]
              edge = cell.partialAlignments_fear[edgeNumber]
              neighbor.append(edge)
            neighborEdge, boundingBox = self.createEdge(neighbor,
                                                        currentNode,
                                                        span)
            neighborEdge.position = neighborPosition
            neighborEdge.fear = (1 - neighborEdge.fscore) + neighborEdge.score
            heappush(queue_fear, (neighborEdge.fear*-1, neighborEdge))

        # FINALIZE
        sortedItems_fear = []
        while(len(currentNode.partialAlignments_fear) > 0):
          (_, obj) = heappop(currentNode.partialAlignments_fear)
          sortedItems_fear.insert(0, obj)
        currentNode.partialAlignments_fear = sortedItems_fear

  def createEdge(self, childEdges, currentNode, span):
    """
    Create a new edge from the list of edges 'edge'.
    Creating an edge involves:
    (1) Initializing the PartialGridAlignment data structure
    (2) Adding links (f,e) to list newEdge.links
    (3) setting the score of the edge with scoreEdge(newEdge, ...)
    In addition, set the score of the new edge.
    """
    newEdge = PartialGridAlignment()
    newEdge.scoreVector_local = svector.Vector()
    newEdge.scoreVector = svector.Vector()

    for e in childEdges:
      newEdge.links += e.links
      newEdge.scoreVector_local += e.scoreVector_local
      newEdge.scoreVector += e.scoreVector

      if e.boundingBox is None:
        e.boundingBox = self.boundingBox(e.links)
    score, boundingBox = self.scoreEdge(newEdge,
                                        currentNode,
                                        span,
                                        childEdges)
    return newEdge, boundingBox

  ############################################################################
  # scoreEdge(self, edge, currentNode, srcSpan, childEdges):
  ############################################################################
  def scoreEdge(self, edge, currentNode, srcSpan, childEdges):
    """
    Score an edge.
    (1) edge: new hyperedge in the alignment forest, tail of this hyperedge are the edges in childEdges
    (2) currentNode: the currentNode in the tree
    (3) srcSpan: span (i, j) of currentNode; i = index of first terminal node in span, j = index of last terminal node in span
    (4) childEdges: the two (or more in case of general trees) nodes we are combining with a new hyperedge
    """

    if self.COMPUTE_ORACLE:
      edge.fscore = self.ff_fscore(edge, srcSpan)

    boundingBox = None
    if self.DO_RESCORE:
      ##################################################################
      # Compute data needed for certain feature functions
      ##################################################################
      tgtSpan = None
      if len(edge.links) > 0:
        boundingBox = self.boundingBox(edge.links)
        tgtSpan = (boundingBox[0][0], boundingBox[1][0])
      edge.boundingBox = boundingBox

      # TODO: This is an awful O(l) patch of code
      linkedIndices = defaultdict(list)
      for link in edge.links:
        fIndex = link[0]
        eIndex = link[1]
        linkedIndices[fIndex].append(eIndex)

      scoreVector = svector.Vector(edge.scoreVector)

      if currentNode.data is not None and currentNode.data is not '_XXX_':
        for _, func in enumerate(self.featureTemplates_nonlocal):
          value_dict = func(self.info, currentNode, edge, edge.links, srcSpan, tgtSpan, linkedIndices, childEdges, self.diagValues, self.treeDistValues)
          for name, value in value_dict.iteritems():
            if value != 0:
              scoreVector[name] = value
      edge.scoreVector = scoreVector

      ##################################################
      # Compute final score for this partial alignment
      ##################################################
      edge.score = edge.scoreVector.dot(self.weights)

    return edge.score, boundingBox

  def boundingBox(self, links):
    """
    Return a 2-tuple of ordered paris representing
    the bounding box for the links in list 'links'.
    (upper-left corner, lower-right corner)
    """
    # upper left corner is (min(fIndices), min(eIndices))
    # lower right corner is (max(fIndices, max(eIndices))

    minF = float('inf')
    maxF = float('-inf')
    minE = float('inf')
    maxE = float('-inf')

    for link in links:
      fIndex = link[0]
      eIndex = link[1]
      if fIndex > maxF:
        maxF = fIndex
      if fIndex < minF:
        minF = fIndex
      if eIndex > maxE:
        maxE = eIndex
      if eIndex < minE:
        minE = eIndex
    # This box is the top-left corner and the lower-right corner
    box = ((minF, minE), (maxF, maxE))
    return box

  def terminal_operation(self, index, currentNode = None):
    """
    Fire features at (pre)terminal nodes of the tree.
    """
    ##################################################
    # Setup
    ##################################################

    partialAlignments = []
    partialAlignments_hope = []
    partialAlignments_fear = []
    oracleAlignment = None

    heapify(partialAlignments)

    tgtWordList = self.f
    srcWordList = self.e
    tgtWord = None
    srcWord = currentNode.children[0].data
    srcTag = currentNode.data
    tgtIndex = None
    srcIndex = currentNode.children[0].eIndex

    span = (srcIndex, srcIndex)

    ##################################################
    # null partial alignment ( assign no links )
    ##################################################
    tgtIndex = -1
    tgtWord = '*NULL*'
    scoreVector = svector.Vector()
    # Compute feature score

    for k, func in enumerate(self.featureTemplates):
      value_dict = func(self.info, tgtWord, srcWord, tgtIndex, srcIndex, [], self.diagValues, currentNode)
      for name, value in value_dict.iteritems():
        if value != 0:
          scoreVector[name] += value

    nullPartialAlignment = PartialGridAlignment()
    nullPartialAlignment.score = score = scoreVector.dot(self.weights)
    nullPartialAlignment.scoreVector = scoreVector
    nullPartialAlignment.scoreVector_local = svector.Vector(scoreVector)

    self.addPartialAlignment(partialAlignments, nullPartialAlignment, self.BEAM_SIZE)

    if self.COMPUTE_ORACLE or self.COMPUTE_FEAR:
      nullPartialAlignment.fscore = self.ff_fscore(nullPartialAlignment, span)

      if self.COMPUTE_ORACLE:
        oracleAlignment = nullPartialAlignment
      if self.COMPUTE_HOPE:
        nullPartialAlignment.hope = nullPartialAlignment.fscore + nullPartialAlignment.score
        self.addPartialAlignment_hope(partialAlignments_hope, nullPartialAlignment, self.BEAM_SIZE)
      if self.COMPUTE_FEAR:
        nullPartialAlignment.fear = (1 - nullPartialAlignment.fscore) + nullPartialAlignment.score
        self.addPartialAlignment_fear(partialAlignments_fear, nullPartialAlignment, self.BEAM_SIZE)

    ##################################################
    # Single-link alignment
    ##################################################
    bestTgtWords = []
    for tgtIndex, tgtWord in enumerate(tgtWordList):
      currentLinks = [(tgtIndex, srcIndex)]
      scoreVector = svector.Vector()

      for k, func in enumerate(self.featureTemplates):
        value_dict = func(self.info, tgtWord, srcWord, tgtIndex, srcIndex, currentLinks, self.diagValues, currentNode)
        for name, value in value_dict.iteritems():
          if value != 0:
            scoreVector[name] += value

      # Keep track of scores for all 1-link partial alignments
      score = scoreVector.dot(self.weights)
      bestTgtWords.append((score, tgtIndex))

      singleLinkPartialAlignment = PartialGridAlignment()
      singleLinkPartialAlignment.score = score
      singleLinkPartialAlignment.scoreVector = scoreVector
      singleLinkPartialAlignment.scoreVector_local = svector.Vector(scoreVector)
      singleLinkPartialAlignment.links = currentLinks

      self.addPartialAlignment(partialAlignments, singleLinkPartialAlignment, self.BEAM_SIZE)

      if self.COMPUTE_ORACLE or self.COMPUTE_FEAR:
        singleLinkPartialAlignment.fscore = self.ff_fscore(singleLinkPartialAlignment, span)

        if self.COMPUTE_ORACLE:
          if singleLinkPartialAlignment.fscore > oracleAlignment.fscore:
            oracleAlignment = singleLinkPartialAlignment

        if self.COMPUTE_HOPE:
          singleLinkPartialAlignment.hope = singleLinkPartialAlignment.fscore + singleLinkPartialAlignment.score
          self.addPartialAlignment_hope(partialAlignments_hope, singleLinkPartialAlignment, self.BEAM_SIZE)

        if self.COMPUTE_FEAR:
          singleLinkPartialAlignment.fear = (1-singleLinkPartialAlignment.fscore)+singleLinkPartialAlignment.score
          self.addPartialAlignment_fear(partialAlignments_fear, singleLinkPartialAlignment, self.BEAM_SIZE)

    ##################################################
    # Two link alignment
    ##################################################
    # Get ready for 2-link alignments

    # Sort the fwords by score
    bestTgtWords.sort(reverse=True)
    LIMIT = max(10, len(bestTgtWords)/2)

    for index1, obj1 in enumerate(bestTgtWords[0:LIMIT]):
      for _, obj2 in enumerate(bestTgtWords[index1+1:LIMIT]):
        # clear contents of twoLinkPartialAlignment
        tgtIndex_a = obj1[1]
        tgtIndex_b = obj2[1]
        # Don't consider a pair (tgtIndex_a, tgtIndex_b) if distance between
        # these indices > 1 (Arabic/English only).
        # Need to debug feature that is supposed to deal with this naturally.
        if self.LANG == "ar_en":
          if (abs(tgtIndex_b - tgtIndex_a) > 1):
            continue

        tgtWord_a = tgtWordList[tgtIndex_a]
        tgtWord_b = tgtWordList[tgtIndex_b]
        currentLinks = [(tgtIndex_a, srcIndex), (tgtIndex_b, srcIndex)]

        scoreVector = svector.Vector()
        for k, func in enumerate(self.featureTemplates):
          value_dict = func(self.info, tgtWord, srcWord,
                            tgtIndex, srcIndex, currentLinks,
                            self.diagValues, currentNode)
          for name, value in value_dict.iteritems():
            if value != 0:
              scoreVector[name] += value

        score = scoreVector.dot(self.weights)

        twoLinkPartialAlignment = PartialGridAlignment()
        twoLinkPartialAlignment.score = score
        twoLinkPartialAlignment.scoreVector = scoreVector
        twoLinkPartialAlignment.scoreVector_local = svector.Vector(scoreVector)
        twoLinkPartialAlignment.links = currentLinks

        self.addPartialAlignment(partialAlignments, twoLinkPartialAlignment, self.BEAM_SIZE)
        if self.COMPUTE_ORACLE or self.COMPUTE_FEAR:
          twoLinkPartialAlignment.fscore = self.ff_fscore(twoLinkPartialAlignment, span)

          if self.COMPUTE_ORACLE:
            if twoLinkPartialAlignment.fscore > oracleAlignment.fscore:
              oracleAlignment = twoLinkPartialAlignment

          if self.COMPUTE_HOPE:
            twoLinkPartialAlignment.hope = twoLinkPartialAlignment.fscore + twoLinkPartialAlignment.score
            self.addPartialAlignment_hope(partialAlignments_hope, twoLinkPartialAlignment, self.BEAM_SIZE)

          if self.COMPUTE_FEAR:
            twoLinkPartialAlignment.fear = (1-twoLinkPartialAlignment.fscore)+twoLinkPartialAlignment.score
            self.addPartialAlignment_fear(partialAlignments_fear, twoLinkPartialAlignment, self.BEAM_SIZE)

    ########################################################################
    # Finalize. Sort model-score list and then hope list.
    ########################################################################
    # Sort model score list.
    sortedBestFirstPartialAlignments = []
    while len(partialAlignments) > 0:
      sortedBestFirstPartialAlignments.insert(0,heappop(partialAlignments))
    # Sort hope score list.
    if self.COMPUTE_HOPE:
      sortedBestFirstPartialAlignments_hope = []
      while len(partialAlignments_hope) > 0:
        (_, obj) = heappop(partialAlignments_hope)
        sortedBestFirstPartialAlignments_hope.insert(0,obj)
    # Sort fear score list.
    if self.COMPUTE_FEAR:
      sortedBestFirstPartialAlignments_fear = []
      while len(partialAlignments_fear) > 0:
        (_, obj) = heappop(partialAlignments_fear)
        sortedBestFirstPartialAlignments_fear.insert(0, obj)

    currentNode.partialAlignments = sortedBestFirstPartialAlignments
    if self.COMPUTE_FEAR:
      currentNode.partialAlignments_fear = sortedBestFirstPartialAlignments_fear
    if self.COMPUTE_HOPE:
      currentNode.partialAlignments_hope = sortedBestFirstPartialAlignments_hope
    if self.COMPUTE_ORACLE:
      currentNode.oracle = None
      # Oracle BEFORE beam is applied
      currentNode.oracle = oracleAlignment

      # Oracle AFTER beam is applied
      #oracleCandidates = list(partialAlignments)
      #oracleCandidates.sort(key=attrgetter('fscore'),reverse=True)
      #currentNode.oracle = oracleCandidates[0]
  ############################################################################
  # addPartialAlignment(self, list, partialAlignment):
  # Add partial alignment to the list of possible partial alignments
  # - Make sure we only keep P partial alignments at any one time
  # - If new partial alignment is > than min(list)
  # - - Replace min(list) with new partialAlignment
  ############################################################################

  def addPartialAlignment(self, list, partialAlignment, BEAM_SIZE):
      # Sort this heap with size limit self.BEAM_SIZE in worst-first order
      # A low score is worse than a higher score

      if len(list) < BEAM_SIZE:
        heappush(list, partialAlignment)
      elif partialAlignment > list[0]:
        heappushpop(list, partialAlignment)

  ############################################################################
  # addPartialAlignment(self, list, partialAlignment):
  # Add partial alignment to the list of possible partial alignments
  # - Make sure we only keep P partial alignments at any one time
  # - If new partial alignment is > than min(list)
  # - - Replace min(list) with new partialAlignment
  ############################################################################

  def addPartialAlignment_hope(self, list, partialAlignment, BEAM_SIZE):
      # Sort this heap with size limit self.BEAM_SIZE in worst-first order
      # A low score is worse than a higher score
      # Use the tuple trick to force Python's
      # heapq to sort by the hope score

      if len(list) < BEAM_SIZE:
        heappush(list,  (partialAlignment.hope, partialAlignment))
      else:
        if partialAlignment.hope > list[0][0]:
          heappushpop(list, (partialAlignment.hope, partialAlignment))

  ############################################################################
  # addPartialAlignment(self, list, partialAlignment):
  # Add partial alignment to the list of possible partial alignments
  # - Make sure we only keep P partial alignments at any one time
  # - If new partial alignment is > than min(list)
  # - - Replace min(list) with new partialAlignment
  ############################################################################

  def addPartialAlignment_fear(self, list, partialAlignment, BEAM_SIZE):
      # Sort this heap with size limit self.BEAM_SIZE in worst-first order
      # A low score is worse than a higher score
      # Use the tuple trick to force Python's heapq to sort by the
      # fear score

      if len(list) < BEAM_SIZE:
        heappush(list,  (partialAlignment.fear, partialAlignment))
      else:
        if partialAlignment.fear > list[0][0]:
          heappushpop(list, (partialAlignment.fear, partialAlignment))

  ############################################################################
  # ff_fscore(self):
  # Compute f-score of an edge wrt the entire gold alignment
  # It shouldn't matter if we compute f-score of an edge wrt the entire
  # alignment or wrt the same piece of the gold alignment. The fscore for the
  # former will just have a lower recall figure.
  ############################################################################

  def ff_fscore(self, edge, span = None):
    if span is None:
      span = (0, len(self.e)-1)

    # get gold matrix span that we are interested in
    # Will be faster than using the matrix operation since getLinksByEIndex
    # returns a sparse list. We also memoize.
    numGoldLinks = self.gold.numLinksInSpan.get(span, None)
    if numGoldLinks is None:
      numGoldLinks = len(self.gold.getLinksByEIndex(span))
      self.gold.numLinksInSpan[span] = numGoldLinks
    else:
      numGoldLinks = self.gold.numLinksInSpan[span]

    # Count our links within this span.
    numModelLinks = len(edge.links)

    # (1) special case: both empty
    if numGoldLinks == 0 and numModelLinks == 0:
      return 1.0
    # (2) special case: gold empty, model not empty OR
    #     gold empty and model not empty
    elif numGoldLinks == 0 or numModelLinks == 0:
      return 0.0

    # The remainder here is executed when numGoldLinks > 0 and
    # numModelLinks > 0

    inGold = self.gold.links_dict.has_key
    numCorrect = 0
    for link in edge.links:
      numCorrect += inGold(link)
    numCorrect = float(numCorrect)

    precision = numCorrect / numModelLinks
    recall = numCorrect / numGoldLinks

    if precision == 0 or recall == 0:
      return 0.0
    f1 = (2*precision*recall) / (precision + recall)
    # Favor recall a la Fraser '07
    # f_recall = 1./((0.1/precision)+(0.9/recall))
    return f1
