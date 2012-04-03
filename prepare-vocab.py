#!/usr/bin/env python
# unigram.py
# riesa@isi.edu (Jason Riesa)
# Build vocabulary file.

import sys
import collections
vcb = collections.defaultdict(int)

for line in sys.stdin:
  words = line.strip().split()
  for word in words:
    vcb[word]+= 1
for word, count in vcb.iteritems():
  print word, count
