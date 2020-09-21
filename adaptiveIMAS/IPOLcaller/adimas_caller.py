#!/usr/bin/env python
import cv2
import sys
sys.path.append(".")
from libLocalDesc import *
from library import *

img1 = cv2.cvtColor(cv2.imread(opt.im1),cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread(opt.im2),cv2.COLOR_BGR2GRAY)

matches = IMAScaller(img1,img2, desc = opt.descriptor, Visual=opt.visual, covering=opt.covering, GFilter='IntCode-'+str(opt.gfilter),AdOPT = opt.type, Detector=opt.detector, dilate=opt.dilate, dir=opt.workdir)