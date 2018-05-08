#!/usr/bin/env python
# functions that read input data or write output data for kaldi

import re
import numpy

def read_feat(infile):
# Read feature file of kaldi
# 03a01Fa [
#   0.000774 -13.194040 -4.706814 7.186832 -3.994959
#                       .
#                       :
#   0.839900 -10.089060 1.265802 -0.286197 -1.198579 ]
#                       
#                       OR
# [
#   0.000774 -13.194040 -4.706814 7.186832 -3.994959
#                       .
#                       :
#   0.839900 -10.089060 1.265802 -0.286197 -1.198579 ]

  finfo='None'
  spk_ary = []
  dflag = 0
  c1 = re.compile(r"(?P<name>\w*)\s*\[")
  c2 = re.compile(r"(?P<data>.*)\]")
  f = open(infile,'r')
  out_data = []
  tmp_ary = []
  for line in f:
    be = c1.match(line) # data begin
    en = c2.match(line) # data end
    if be :
      if be.group('name'):
        finfo = be.group('name')
      else :
        finfo = 'None'
      dflag = 1
      spk_ary.append(finfo)
      tmp_ary = []
    elif en:
      dflag = 0
      line = en.group('data')
      tmp_ary.append(line.split())
      out_data.append(tmp_ary)
    elif dflag:
      tmp_ary.append(line.split())

  f.close()

  return spk_ary, out_data
    


