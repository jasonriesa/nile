#!/usr/bin/env python
# riesa@isi.edu (Jason Riesa)
# Training and Alignment
###############################################
# Based on work described in:
###############################################
#@inproceedings{RiesaIrvineMarcu:11,
#  Title = {Feature-Rich Language-Independent Syntax-Based Alignment for Statistical Machine Translation},
#  Author = {Jason Riesa and Ann Irvine and Daniel Marcu},
#  Booktitle = {Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
#  Pages = {497--507},
#  Publisher = {Association for Computational Linguistics},
#  Year = {2011}}
#
# @inproceedings{RiesaMarcu:10,
#   Title = {Hierarchical Search for Word Alignment},
#   Author = {Jason Riesa and Daniel Marcu},
#   Booktitle = {Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics (ACL)},
#   Pages = {157--166},
#   Publisher = {Association for Computational Linguistics},
#   Year = {2010}}
###############################################

from collections import defaultdict
from itertools import izip
import cPickle
import os
import sys
import random
import tempfile
import time

import Alignment
import Fmeasure
import GridAlign
import gflags as flags
import io_helper
import mpi
import svector
from pyglog import *

FLAGS = flags.FLAGS

def readWeights(weights_file):
  """
  Read feature function weights from an input file.
  This function reads a pickled svector object.
  """
  return cPickle.load(weights_file)

def robustRead(filename):
  """
  A wrapper for more robust opening of files for reading.
  """
  success = False
  filehandle = None
  attempt_count = 0
  while (not success) and (attempt_count < 10):
    try:
      attempt_count += 1
      filehandle = open(filename, 'r')
      success = True
    except:
      time.sleep(10)
  if attempt_count >= 10:
    LOG(FATAL, "Could not open file %s for reading. Attempted 10 times." % (filename))
  return filehandle

def robustWrite(filename):
  """
  A wrapper for more robust opening of files for writing.
  """
  success = False
  filehandle = None
  attempt_count = 0
  while (not success) and (attempt_count < 10):
    try:
      attempt_count += 1
      filehandle = open(filename, 'w')
      success = True
    except:
      time.sleep(10)
  if attempt_count >= 10:
    LOG(FATAL, "Could not open file %s for writing. Attempted 10 times." % (filename))
  return filehandle

def readVocab(infile):
  """ Read vocabulary from an input file, line by line.
  Used later for other tasks, like data filtering. """
  # Read floating-point numbers line by line
  # Q: Do boolean values take up less space than integers?
  vcb = { }
  # Add the null token to the vocabulary
  vcb['*NULL*'] = True
  for line in infile:
    word = line.strip().split()[0]
    vcb[word] = True
    try:
      word_utf8 = word.decode('utf-8')
    except:
      continue
  infile.close()
  return vcb

def readPef(file, e_vcb, f_vcb):
  """
  Read a P(e|f) table in format:
  <e>  <f>  <p(e|f)>
  """
  pef = defaultdict(dict)
  for line in file:
    (eword, fword, prob) = line.split()
    # Filter by vocabulary
    # Do not store tuples that we will never use
    if e_vcb.has_key(eword) and f_vcb.has_key(fword):
      pef[fword][eword] = float(prob)
  file.close()
  return pef

def readPfe(file, e_vcb, f_vcb):
  """
  Read a P(f|e) table in format:
  <f>  <e>  <p(f|e)>
  """
  pfe = defaultdict(dict)
  for line in file:
    (fword, eword, prob) = line.split()
    # Filter by vocabulary
    # Do not store tuples that we will never use
    if e_vcb.has_key(eword) and f_vcb.has_key(fword):
      pfe[eword][fword] = float(prob)
  file.close()
  return pfe

def decode_parallel(weights, indices, blob, name="", out=sys.stdout):
  """
  Align some input data in blob with a given weight vector. Report accuracy.
  """
  myRank = mpi.rank
  masterRank = 0
  # How many processors are there?
  nProcs = mpi.size

  results = [ ]
  allResults = None
  fmeasure = 0.0

  ##########################################
  # Keep track of time to train this epoch
  ##########################################
  startTime = time.time()
  result_file = robustWrite(tmpdir+'/results.'+str(mpi.rank))

  for i, instanceID in enumerate(indices[:FLAGS.subset]):
    if myRank == i % nProcs:
      # Assign the current instances we will look at
      f = blob['f_instances'][instanceID]
      e = blob['e_instances'][instanceID]
      etree = blob['etree_instances'][instanceID]
      if FLAGS.train:
        gold_str = blob['gold_instances'][instanceID]
        gold = Alignment.Alignment(gold_str)

      ftree = None
      if FLAGS.ftrees is not None:
        ftree = blob['ftree_instances'][instanceID]

      inverse = None
      if FLAGS.inverse is not None:
        inverse = blob['inverse_instances'][instanceID]

      a1 = None
      if FLAGS.a1 is not None:
        a1 = blob['a1_instances'][instanceID]

      a2 = None
      if FLAGS.a2 is not None:
        a2 = blob['a2_instances'][instanceID]

      # Prepare input data.
      # f, e are sequences of words
      f = f.split()
      e = e.split()

      # Initialize model for this instance
      model = GridAlign.Model(f, e, etree, ftree, instanceID, weights, a1, a2,
                              inverse, DECODING=True,
                              LOCAL_FEATURES=blob['localFeatures'],
                              NONLOCAL_FEATURES=blob['nonlocalFeatures'],
                              FLAGS=FLAGS)
      if FLAGS.train:
        model.gold = gold
      # Initialize model with data tables
      model.pef = blob['pef']
      model.pfe = blob['pfe']
      # Align the current training instance
      # FOR PROFILING: cProfile.run('model.align(1)','profile.out')
      model.align()
      # Dump intermediate chunk to disk. Reassemble later.
      if FLAGS.train:
        cPickle.dump((model.modelBest.links, model.gold.links_dict), result_file, protocol=cPickle.HIGHEST_PROTOCOL)
      elif FLAGS.align:
        cPickle.dump(model.modelBest.links, result_file, protocol=cPickle.HIGHEST_PROTOCOL)

  result_file.close()
  done = mpi.gather(value=True, root=0)

  # REDUCE HERE
  if myRank == masterRank:
    # Open result files for reading
    resultFiles = { }
    for i in range(nProcs):
      resultFiles[i] = open(tmpdir+'/results.'+str(i),'r')

    if FLAGS.train:
      ##########################################################################
      # Compute f-measure over all alignments
      ##########################################################################
      numCorrect = 0
      numModelLinks = 0
      numGoldLinks = 0

      for i, instanceID in enumerate(indices[:FLAGS.subset]):
        # What node stored instance i
        node = i % nProcs
        # Retrieve result from instance i
        resultTuple = cPickle.load(resultFiles[node])
        modelBest = resultTuple[0]
        gold = resultTuple[1]
        # Update F-score counts
        numCorrect_, numModelLinks_, numGoldLinks_ = f1accumulator(modelBest,
                                                                   gold)
        numCorrect += numCorrect_
        numModelLinks += numModelLinks_
        numGoldLinks += numGoldLinks_
      # Compute F-measure, Precision, and Recall
      fmeasure, precision, recall = f1score(numCorrect,
                                            numModelLinks,
                                            numGoldLinks)
      elapsedTime = time.time() - startTime

      ######################################################################
      # Print report for this iteration
      ######################################################################
      sys.stderr.write("Time: "+str(elapsedTime)+"\n")
      sys.stderr.write("\n")
      sys.stderr.write('F-score-%s: %1.5f\n' % (name, fmeasure))
      sys.stderr.write('Precision-%s: %1.5f\n' % (name, precision))
      sys.stderr.write('Recall-%s: %1.5f\n' % (name, recall))
      sys.stderr.write('# Correct: %d\n' % (numCorrect))
      sys.stderr.write('# Me Total: %d\n' % (numModelLinks))
      sys.stderr.write('# Gold Total: %d\n' % (numGoldLinks))
      sys.stderr.write("[%d] Finished decoding.\n" %(myRank))
    else:
      for i, instanceID in enumerate(indices):
        node = i % nProcs
        modelBestLinks = cPickle.load(resultFiles[node])
        out.write("%s\n" %(" ".join(map(lambda link: "%s-%s" %(link[0], link[1]), modelBestLinks))))
    # CLEAN UP
    for i in range(nProcs):
      resultFiles[i].close()
  return

def perceptron_parallel(epoch, indices, blob, weights = None, valid_feature_names = None):
  """
  Implements parallelized version of perceptron training for structured outputs
  (Collins, 2002; McDonald, 2010).
  """
  # Which processor am I?
  myRank = mpi.rank
  # Let processor 0 be the master.
  masterRank = 0
  # How many processors are there?
  nProcs = mpi.size
  ##########################################
  # Keep track of time to train this epoch
  ##########################################
  startTime = time.time()
  # Restart with weights from last epoch or 0.
  # Will ignore any weights passed during function call.
  weights_restart_filename = '%s/training-restart.%s' % (tmpdir, str(mpi.rank))
  if os.path.isfile(weights_restart_filename):
    weights_restart_file = open(weights_restart_filename, 'r')
    weights = cPickle.load(weights_restart_file)
    weights_restart_file.close()
  else:
    # If weights passed during function call is None start with empty.
    if weights is None or len(weights) == 0:
        weights = svector.Vector()

  # Restart with previous running weight sum, also.
  weights_sum_filename = '%s/training.%s' % (tmpdir, str(mpi.rank))
  if os.path.isfile(weights_sum_filename):
    weights_sum_file = open(weights_sum_filename, 'r')
    weights_sum = cPickle.load(weights_sum_file)
    weights_sum_file.close()
  else:
    weights_sum = svector.Vector()

  numChanged = 0
  done = False
  for i, instanceID in enumerate(indices[:FLAGS.subset]):
    if myRank == i % nProcs:
      # Assign the current instances we will look at
      f = blob['f_instances'][instanceID]
      e = blob['e_instances'][instanceID]
      etree = blob['etree_instances'][instanceID]
      gold_str = blob['gold_instances'][instanceID]

      inverse = None
      if FLAGS.inverse is not None:
        inverse = blob['inverse_instances'][instanceID]

      a1 = None
      if FLAGS.a1 is not None:
        a1 = blob['a1_instances'][instanceID]

      a2 = None
      if FLAGS.a2 is not None:
        a2 = blob['a2_instances'][instanceID]

      ftree = None
      if FLAGS.ftrees is not None:
        ftree = blob['ftree_instances'][instanceID]

      # Preprocess input data
      # f, e are sequences of words
      f = f.split() ; e = e.split()

      # gold is a sequence of f-e link pairs
      gold = Alignment.Alignment(gold_str)

      # Initialize model for this instance
      model = GridAlign.Model(f, e, etree, ftree, instanceID, weights, a1, a2,
                              inverse, LOCAL_FEATURES=blob['localFeatures'],
                              NONLOCAL_FEATURES=blob['nonlocalFeatures'],
                              FLAGS=FLAGS)
      model.gold = gold

      # Initialize model with data tables
      model.pef = blob['pef']
      model.pfe = blob['pfe']
      # Align the current training instance
      model.align()

      ######################################################################
      # Weight updating
      ######################################################################
      LEARNING_RATE = FLAGS.learningrate

      # Set the oracle item
      oracle = None
      if FLAGS.oracle == 'gold':
        oracle = model.oracle
      elif FLAGS.oracle == 'hope':
        oracle = model.hope
      else:
        sys.stderr.write("ERROR: Unknown oracle class: %s\n" %(FLAGS.oracle))

      # Set the hypothesis item
      hyp = None
      if FLAGS.hyp == '1best':
        hyp = model.modelBest
      elif FLAGS.hyp == 'fear':
        hyp = model.fear
      else:
        sys.stderr.write("ERROR: Unknown hyp class: %s\n" %(FLAGS.hyp))

      # Debiasing
      if FLAGS.debiasing:
        validate_features(oracle.scoreVector, valid_feature_names)
        validate_features(hyp.scoreVector, valid_feature_names)

      deltas = None
      if set(hyp.links) != set(oracle.links):
        numChanged += 1
        ###############################################################
        # WEIGHT UPDATES
        ################################################################
        deltas = oracle.scoreVector - hyp.scoreVector
        weights = weights + LEARNING_RATE*deltas
      # Even if we didnt update, the current weight vector should count towards the sum!
      weights_sum += weights
      # L1 Projection step
      # if w in [-tau, tau], w -> 0
      # else, move w closer to 0 by tau.
      if FLAGS.tau is not None:
        for index, w in weights_sum.iteritems():
          if w == 0:
            del weights_sum[index]
            continue
          if index[-3:] == '_nb':
            continue
          if w > 0 and w <= FLAGS.tau and not FLAGS.negreg:
            del weights_sum[index]
          elif w < 0 and w >= (FLAGS.tau * -1):
            del weights_sum[index]
          elif w > 0 and w > FLAGS.tau and not FLAGS.negreg:
            weights_sum[index] -= FLAGS.tau
          elif w < 0 and w < (FLAGS.tau * -1):
            weights_sum[index] += FLAGS.tau

  # Set uniq pickled output file for this process
  # Holds sum of weights over each iteration for this process
  output_filename = "%s/training.%s" %(tmpdir, str(mpi.rank))
  output_file = open(output_filename,'w')
  # Dump all weights used during this node's run; to be averaged by master along with others
  cPickle.dump(weights_sum, output_file, protocol=cPickle.HIGHEST_PROTOCOL)
  output_file.close()

  # Remeber just the last weights used for this process; start here next epoch.
  output_filename_last_weights = "%s/training-restart.%s" %(tmpdir, str(mpi.rank))
  output_file_last_weights = open(output_filename_last_weights,'w')
  cPickle.dump(weights, output_file_last_weights, protocol=cPickle.HIGHEST_PROTOCOL)
  output_file_last_weights.close()

  #############################################
  # Gather "done" messages from workers
  #############################################
  # Synchronize
  done = mpi.gather(value=True,root=0)

  #####################################################################################
  # Compute f-measure over all alignments
  #####################################################################################
  masterWeights = svector.Vector()

  if myRank == masterRank:
    # Read pickled output
    for rank in range(nProcs):
      input_filename = tmpdir+'/training.'+str(rank)
      input_file = open(input_filename,'r')
      masterWeights += cPickle.load(input_file)
      input_file.close()
    sys.stderr.write("Done reading data.\n")
    sys.stderr.write("len(masterWeights)= %d\n"%(len(masterWeights)))
    sys.stderr.flush()

    ######################################################
    # AVERAGED WEIGHTS
    ######################################################
    sys.stderr.write("[%d] Averaging weights.\n" %(mpi.rank))
    sys.stderr.flush()
    masterWeights = masterWeights / (len(indices) * (epoch+1))
    # Dump master weights to file
    # There is only one weight vector in this file at a time.
    mw = robustWrite(tmpdir+'/weights')
    cPickle.dump(masterWeights,mw,protocol=cPickle.HIGHEST_PROTOCOL)
    mw.close()

  ######################################################################
  # All processes read and load new averaged weights
  ######################################################################
  # But make sure worker nodes don't attempt to read from the weights
  # file before the root node has written it.
  # Sync-up with a blocking broadcast call
  ready = mpi.broadcast(value=True, root=0)
  mw = robustRead(tmpdir+'/weights')
  masterWeights = cPickle.load(mw)
  mw.close()

  ######################################################################
  # Print report for this iteration
  ######################################################################
  elapsedTime = time.time() - startTime
  if myRank == masterRank:
    # masterRank is printing elapsed time.
    # May differ at each node.
    sys.stderr.write("Time: %0.2f\n" %(elapsedTime))
    sys.stderr.write("[%d] Finished training.\n" %(mpi.rank))

  return masterWeights

def f1score(numCorrect, numModelLinks, numGoldLinks):
  if numGoldLinks == 0 and numModelLinks == 0:
    return 1.0, 1.0, 1.0
  elif numGoldLinks == 0 or numModelLinks == 0:
    return 0.0, 0.0, 0.0

  precision = numCorrect / numModelLinks
  recall = numCorrect / numGoldLinks

  if precision == 0 or recall == 0:
    return 0.0, 0.0, 0.0

  f1 = (2*precision*recall) / (precision + recall)
  return f1, precision, recall

def f1accumulator(hyp, gold):
  numModelLinks = len(hyp)
  numGoldLinks = len(gold)

  if numGoldLinks == 0 and numModelLinks == 0:
    return 0.0, numModelLinks, numGoldLinks
  elif numGoldLinks == 0 or numModelLinks == 0:
    return 0.0, numModelLinks, numGoldLinks

  numCorrect = 0.0
  for link in hyp:
    numCorrect += link in gold
  return numCorrect, numModelLinks, numGoldLinks

def validate_features(weights, valid_feature_names):
  """
  Get rid of features not in valid_feature_names.
  """
  for k in weights.iterkeys():
    if not valid_feature_names.has_key(k):
      del weights[k]

def getFeatureNames(weights):
  """
  Get feature names (keys) from an input svector object.
  Return as a hashtable for quick lookup later.
  """
  valid_feature_names = { }
  for k in weights.iterkeys():
    valid_feature_names[k] = True
  return valid_feature_names

def validateInput(FLAGS):
  """
  Validate input arguments. Terminate with message on error.
  """
  try:
    if not (FLAGS.train ^ FLAGS.align): # xor
      raise Exception, "You must specify one and only one of --train or --align."
    if FLAGS.train and (FLAGS.f is None or FLAGS.e is None or FLAGS.etrees is None or FLAGS.gold):
      raise Exception, "Not all required arguments properly specified."
    if FLAGS.train and (FLAGS.fdev is None or FLAGS.edev is None or FLAGS.etreesdev is None or FLAGS.golddev is None):
      raise Exception, "No heldout devset provided for training."
  except Exception, msg:
    if mpi.rank == 0:
      sys.stderr.write("Error: %s\nSee %s --help\n" % (msg, sys.argv[0]))
    sys.exit(1)

def do_training(indices, training_blob, heldout_blob, weights, weights_out, debiasing_weights):
  """
  Helper/wrapper function for parallel perceptron training.
  Runs one epoch of perceptron training and reports current accuracy on
  training data and on heldout data.
  """
  # Under de-biasing mode, we only allow features present in a given initial
  # weight vector. These are features that have been "selected" under a previously
  # run regularized training scheme.
  valid_feature_names = None
  if FLAGS.debiasing:
    valid_feature_names = getFeatureNames(debiasing_weights)

  for epoch in range(FLAGS.maxepochs):
    # Randomize order of examples; broadcast this randomized order to all processes.
    # The particular subset any perceptron process gets for this epoch is dependent
    # upon this randomized ordering.
    if myRank == 0 and FLAGS.shuffle:
      random.shuffle(indices)
    indices = mpi.broadcast(value=indices, root=0)

    ##################################################
    # SEARCH: Find 1-best under current model
    ##################################################
    # Run one epoch over training data
    io_helper.write_master("===EPOCH %d TRAINING===\n" %(epoch))
    newWeights_avg = perceptron_parallel(epoch, indices, training_blob, weights,
                                         valid_feature_names)
    ####################################
    # Dump weights for this iteration
    ####################################
    if myRank == 0:
      cPickle.dump(newWeights_avg, weights_out, protocol=cPickle.HIGHEST_PROTOCOL)
 	    # Need to flush output somehow here. Does weights_out.flush() work?
      weights_out.flush()

    ##################################################
    # Try a corpus re-decode here with the new weights
    # This returns a HELDOUT F-SCORE
    ##################################################
    # Decode dev data with same new learned weight vector
    if FLAGS.decodeheldout:
      io_helper.write_master("===EPOCH %d DECODE HELDOUT===\n" %(epoch))
      decode_parallel(newWeights_avg, indices_dev, heldout_blob, "dev")
  if myRank == 0:
    weights_out.close()

if __name__ == "__main__":
    myRank = mpi.rank
    flags.DEFINE_string('f',None,'f training file')
    flags.DEFINE_string('e',None,'e training file')
    flags.DEFINE_string('etrees',None,'etrees training file')
    flags.DEFINE_string('ftrees',None,'ftrees training file')
    flags.DEFINE_string('weights',None,'weights file')
    flags.DEFINE_string('gold',None,'gold training alignments file in f-e format')
    flags.DEFINE_string('fvcb',None,'f vocab file')
    flags.DEFINE_string('evcb',None,'e vocab file')
    flags.DEFINE_string('a1',None,'Third-party alignments in f-e format.')
    flags.DEFINE_string('a2',None,'Third-party alignments in f-e format.')
    flags.DEFINE_string('inverse',None,'f-e inverse alignments (from bottom-up search on foreign tree)')
    flags.DEFINE_integer('init_k',None,'k = initialization beam size')
    flags.DEFINE_integer('k',1,'k = standard beam size')
    flags.DEFINE_integer('maxepochs',100,'maximum number of epochs to run training')
    flags.DEFINE_string('fdev',None,'f heldout file')
    flags.DEFINE_string('edev',None,'e heldout file')
    flags.DEFINE_string('golddev',None,'gold dev alignments file file in f-e format')
    flags.DEFINE_string('etreesdev',None,'etrees dev file')
    flags.DEFINE_string('ftreesdev',None,'ftrees dev file')
    flags.DEFINE_string('a1_dev',None,'Third-party alignments in f-e format for heldout data')
    flags.DEFINE_string('a2_dev',None,'Third-party alignments in f-e format for heldout data')
    flags.DEFINE_string('inverse_dev',None,'f-e inverse alignments (from bottom-up search on foreign tree)')
    flags.DEFINE_string('srctags',None,'srctags file')
    flags.DEFINE_string('langpair',None,'tell Nile what language-pair it is working on (mostly for importing specific feature sets); default: None')
    flags.DEFINE_string('pef',None,'p(e|f) file')
    flags.DEFINE_string('pfe',None,'p(f|e) file')
    flags.DEFINE_float('learningrate',1.0,'learning rate parameter for perceptron training; default: 1.0')
    flags.DEFINE_string('hyp','1best','hypothesis to compare with oracle. one of {fear, 1best}; default: 1best')
    flags.DEFINE_string('oracle','gold','type of oracle. one of {gold, hope}; default: gold')
    flags.DEFINE_string('weights_out',None,'output file for weights')
    flags.DEFINE_boolean('rescore',True,'True: do rescoring during bottom-up search; False: use only scores at initialization to determine 1best. Default: True')
    flags.DEFINE_boolean('train', False, 'Run discriminative training')
    flags.DEFINE_boolean('align', False, 'Align data with parameters from --weights')
    flags.DEFINE_boolean('decodeheldout',True,'Align heldout data with new weight vector after each epoch.')
    flags.DEFINE_boolean('shuffle',True,'Randomize training instances for each epoch.')
    flags.DEFINE_string('notes','','Any extra notes for this training run')
    flags.DEFINE_boolean('source', False, 'Search bottom-up on the source trees.')
    flags.DEFINE_boolean('target', True, 'Search bottom-up on the target trees.')
    flags.DEFINE_string('out',None,'Output file for alignments in --align mode')
    flags.DEFINE_boolean('skipbadtrees',True,'Skip trees w/o parses')
    flags.DEFINE_integer('subset', None, 'Read only the first k training, dev examples')
    flags.DEFINE_float('tau', None, 'L1 coefficient')
    flags.DEFINE_float('tau_nb', None, 'L1 coefficient for non-binary features')
    flags.DEFINE_boolean('negreg', False, 'Only regularize negative weights')
    flags.DEFINE_boolean('debiasing', False, 'Training under de-biasing mode')
    flags.DEFINE_string('debiasing_weights', None, 'Features to use under de-biasing mode')
    flags.DEFINE_string('tempdir', None, 'User-defined directory location for temporary files')
    argv = FLAGS(sys.argv)

    if FLAGS.debiasing and FLAGS.debiasing_weights is None:
      LOG(FATAL, "Must provide weight vector to use when debiasing mode enabled.")
    if FLAGS.debiasing and FLAGS.tau is not None:
      LOG(FATAL, "Regularization not permitted under debiasing mode. Disable the --tau flag.")

    ##################################################
    # Import features for the specified language-pair
    ##################################################
    # To use language specific features for, e.g. Arabic-English,
    # copy the generic Features.py module to a file called
    # Features_ar_en.py and add your new feature functions
    # functions to the file. Then, just call nile with flag:
    # --langpair ar_en
    # This will cause Nile to load module Features_ar_en.py
    # instead of the standard Features.py
    #
    if FLAGS.langpair is not None:
      try:
        if myRank == 0:
          LOG(INFO, "Language pair %s specified; loading %s featureset." %(FLAGS.langpair, FLAGS.langpair))
        Features = __import__("Features_%s" % (FLAGS.langpair))
      except:
        if myRank == 0:
          err_msg = "Could not import language-specific features Features_%s.py. " %(FLAGS.langpair)
          err_msg += "Using standard featureset."
          LOG(INFO, err_msg)
        import Features
    else:
      import Features

    pid = str(os.getpid())
    if myRank == 0:
      print os.getpid()
      print "NOTES: %s" %(FLAGS.notes)
    file_handles = io_helper.open_files(FLAGS)

    # Use to filter pef/pfe data
    e_vcb = readVocab(file_handles['evcb'])
    f_vcb = readVocab(file_handles['fvcb'])

    #######################################################
    # Load pef
    #######################################################
    pef = readPef(file_handles['pef'], e_vcb, f_vcb)
    file_handles['pef'].close()

    #######################################################
    # Load pfe
    # col1 - e; col2 - f; col3 - count; col4 - prob p(f|e)
    #######################################################
    pfe = readPfe(file_handles['pfe'], e_vcb, f_vcb)
    file_handles['pfe'].close()

    ########################################################
    # Initialize Featureset
    ########################################################
    localFeatures = Features.LocalFeatures(pef, pfe)
    nonlocalFeatures = Features.NonlocalFeatures(pef, pfe)

    e_instances = []
    f_instances = []
    etree_instances = []
    ftree_instances = []
    a1_instances = []
    a2_instances = []
    gold_instances = []
    inverse_instances = []

    if FLAGS.train:
      f_dev_instances = []
      e_dev_instances = []
      etree_dev_instances = []
      ftree_dev_instances = []
      a1_dev_instances = []
      a2_dev_instances = []
      gold_dev_instances = []
      inverse_dev_instances = []

    tmpdir = None
    if mpi.rank == 0:
      base_tempdir = None
      if FLAGS.tempdir is not None:
        base_tempdir = FLAGS.tempdir
      else:
        base_tempdir = tempfile.gettempdir()
      if base_tempdir is None:
        base_tempdir = "."
      tmpdir = tempfile.mkdtemp(prefix='align-'+str(os.getpid())+'-',
                                dir=base_tempdir)
    tmpdir = mpi.broadcast(value=tmpdir, root=0)


    ################################################
    # Load training examples
    ################################################
    for f, e, etree in izip(file_handles['f'],
                               file_handles['e'],
                               file_handles['etrees']):
      f_instances.append(f.strip())
      e_instances.append(e.strip())
      etree_instances.append(etree.strip())
    indices = range(len(e_instances))
    ################################################
    # Load held-out dev examples
    ################################################
    if FLAGS.train:
      for g in file_handles['gold']:
        gold_instances.append(g.strip())
      for f, e, etree, g in izip(file_handles['fdev'],
                                 file_handles['edev'],
                                 file_handles['etreesdev'],
                                 file_handles['golddev']):
        f_dev_instances.append(f.strip())
        e_dev_instances.append(e.strip())
        etree_dev_instances.append(etree.strip())
        gold_dev_instances.append(g.strip())
      indices_dev = range(len(e_dev_instances))

    ################################################
    # LOAD OPTIONAL EXTRAS
    ################################################
    if FLAGS.ftrees is not None:
      for ftree in file_handles['ftrees']:
        ftree_instances.append(ftree.strip())
      if FLAGS.train:
        for ftree in file_handles['ftreesdev']:
          ftree_dev_instances.append(ftree.strip())

    if FLAGS.inverse is not None:
        for inverse in file_handles['inverse']:
          inverse_instances.append(inverse.strip())
        if FLAGS.train:
            for inverse in file_handles['inverse_dev']:
              inverse_dev_instances.append(inverse.strip())

    if FLAGS.a1 is not None:
      for a1 in file_handles['a1']:
        a1_instances.append(a1.strip())
      if FLAGS.train:
        for a1 in file_handles['a1_dev']:
          a1_dev_instances.append(a1.strip())

    if FLAGS.a2 is not None:
      for a2 in file_handles['a2']:
        a2_instances.append(a2.strip())
      if FLAGS.train:
        for a2 in file_handles['a2_dev']:
          a2_dev_instances.append(a2.strip())


    ###########################################################
    # Initialize weights
    ###########################################################
    if FLAGS.weights is not None:
      # Restart from another parameter vector
      weights = readWeights(file_handles['weights'])
    else:
      # Start with empty weight vector
      weights = None

    debiasing_weights = None
    if FLAGS.debiasing_weights is not None:
      debiasing_weights_file = open(FLAGS.debiasing_weights, "r")
      debiasing_weights = readWeights(debiasing_weights_file)

    # Rank 0 is the master node
    # It will delegate to other nodes and collect processed information
    weights_out = None
    if myRank == 0 and FLAGS.train:
      if FLAGS.weights_out is not None:
        weights_out = open(FLAGS.weights_out, 'w')
        print FLAGS.weights_out
      else:
        weights_out = open("weights."+pid, "w")
    ###########################################################
    # Initialize blobs to pass to training and decoding methods
    ###########################################################
    common_blob = {
      'pef': pef,
      'pfe': pfe,
      'localFeatures': localFeatures,
      'nonlocalFeatures': nonlocalFeatures,
      'tmpdir': tmpdir
    }
    training_blob = {
      'f_instances': f_instances,
      'e_instances': e_instances,
      'etree_instances': etree_instances,
      'ftree_instances': ftree_instances,
      'gold_instances': gold_instances,
      'a1_instances': a1_instances,
      'a2_instances': a2_instances,
      'inverse_instances': inverse_instances
    }
    if FLAGS.train:
      heldout_blob = {
        'f_instances': f_dev_instances,
        'e_instances': e_dev_instances,
        'etree_instances': etree_dev_instances,
        'ftree_instances': ftree_dev_instances,
        'gold_instances': gold_dev_instances,
        'a1_instances': a1_dev_instances,
        'a2_instances': a2_dev_instances,
        'inverse_instances': inverse_dev_instances
      }
    training_blob.update(common_blob)
    if FLAGS.train:
      heldout_blob.update(common_blob)

    if FLAGS.train:
      do_training(indices, training_blob, heldout_blob, weights, weights_out, debiasing_weights)
    elif FLAGS.align:
      decode_parallel(weights, indices, training_blob, "align",
                      out=file_handles['out'])
