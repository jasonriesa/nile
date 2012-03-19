#!/usr/bin/env python
# Jason Riesa <riesa@isi.edu>
# 9/12/2008
''' Calculate fmeasure and related figures, given two Alignment objects '''

import sys
from collections import defaultdict
from itertools import izip_longest, izip

class Fmeasure:
    
    def __init__(self):
        self.correct = 0
        self.numMeTotal = 0
        self.numGoldTotal = 0
        
    def accumulate(self, me, gold):
        #Accumulate counts

        meLinks = me.strip().split()
        goldLinks = gold.strip().split()
        
        self.numMeTotal += len(meLinks)
        self.numGoldTotal += len(goldLinks)
        
        goldLinksDict = dict(izip_longest(goldLinks, [None]))
        for link in meLinks:
            if link in goldLinksDict:
                self.correct += 1.0
                
    def accumulate_o(self, edge, goldmatrix):
        
        self.numMeTotal += len(edge.links)
        goldIndices = nonzero(goldmatrix)
        self.numGoldTotal += len(goldIndices[0])
        
        for link in edge.links:
            if goldmatrix[link[0]][link[1]] == 1:
                self.correct += 1.0
        
    '''
    def accumulate_m(self, model, gold):
        # model and gold are matrices, each representing
        # links in an alignment; a cell holding '1' denotes a link
        
        self.numMeTotal += sum(model)
        self.numGoldTotal += sum(gold)
        # sum matrices together; cells that hold a '2' are correct links
        self.correct += len(nonzero((model+gold)==2)[0])
    '''    
    
    def report(self):
        ''' Report f-score and related figures '''
        
        precision = self.precision() 
        recall = self.recall()
        if (precision + recall) == 0:
            fscore = 0.0
        else:
            fscore = (2.0*precision*recall)/(precision + recall)
        
        #fscore = self.f1score()
        
        sys.stdout.write('F-score: %1.5f\n' % (fscore))
        sys.stdout.write('Precision: %1.5f\n' % (precision))
        sys.stdout.write('Recall: %1.5f\n' % (recall))
        sys.stdout.write('# Correct: %d\n' % (self.correct))
        sys.stdout.write('# Hyp Total: %d\n' % (self.numMeTotal))
        sys.stdout.write('# Gold Total: %d\n' % (self.numGoldTotal))
        return fscore
    
    def precision(self):
        if self.numMeTotal == 0 and self.numGoldTotal == 0:
            return 1.0
        elif self.numMeTotal == 0 or self.numGoldTotal == 0:
            return 0.0
        else:
            return float(self.correct)/self.numMeTotal
        
    def recall(self):
        if self.numMeTotal == 0 and self.numGoldTotal == 0:
            return 1.0
        elif self.numMeTotal == 0 or self.numGoldTotal == 0:
            return 0.0
        else:
            return float(self.correct)/self.numGoldTotal
        
    def f1score(self):
        prec = self.precision() 
        rec = self.recall()
        if prec + rec == 0:
            return 0.0
        else:
            return 2.0*prec*rec/(prec + rec)
    
    def reset(self):
        self.correct = 0.0
        self.numGoldTotal = 0.0
        self.numMeTotal = 0.0
        
        
def score(file1, file2):
         
    fmeasure = Fmeasure()
    for (me_str, gold_str) in izip(open(file1, 'r'), open(file2, 'r')):
        fmeasure.accumulate(me_str, gold_str)
    fmeasure.report()
        

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("Syntax: %s <ALIGNMENT> <GOLD_ALIGNMENT>\n" % (sys.argv[0]))
        sys.exit(1)
    score(sys.argv[1], sys.argv[2])
    
    
