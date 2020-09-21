# When default GPU is being used... prepare to use a second one
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from library import *
import time

import sys
# sys.path.append("hesaffnet")
sys.path.append(opt.bindir+"hesaffnet")
from hesaffnet import *


import subprocess
import pandas
def IMAScaller(img1,img2, desc = 11, AdOPT = None, covering=-1.0, GFilter='IntCode-2', Visual=False, EuPrecision=24, Detector='HessAff', dilate=False, dir='/tmp/'):    
    if GFilter[0:7] == 'IntCode':
        GFilterCode = int(GFilter[8])
    else:
        GFilterCode = 0        

    # example: cmdflag = 'OMP_NUM_THREADS=1'
    cmdflag = ''

    cv2.imwrite("%simg1.png" % dir ,img1)
    cv2.imwrite("%simg2.png" % dir,img2)
    _ = subprocess.check_output("cp imas_bin %s" % dir, shell=True)
    _ = subprocess.call("rm %s/data_matches.csv" % dir, shell=True)
    ET_Ad = 0.0
    simus1, simus2 = [], []
    if AdOPT is not None:
        if Detector=='HessAff':
            start_time = time.time()        
            _, Aq_list, _ = HessAffNetHardNet_Detect(img1,Nfeatures=300)
            _, At_list, _ = HessAffNetHardNet_Detect(img2,Nfeatures=300)
            if dilate:
                dilationSet = [affine_decomp2affine(A) for A in GetCovering(tilts_1_25, endAngle=2*np.pi)]
                Aq_list = [ComposeAffineMaps(A, B) for A in dilationSet for B in Aq_list]
                # No need to do the line below as the dilation set is already covering for 1.7 instead of sqrt(1.7)
                # At_list = [ComposeAffineMaps(A, B) for A in dilationSet for B in At_list]
            ET_Ad = time.time() - start_time
    
            Aq_list = [affine_decomp(cv2.invertAffineTransform(A),ModRots=True) for A in Aq_list] 
            At_list = [affine_decomp(cv2.invertAffineTransform(A),ModRots=True) for A in At_list]
        elif Detector=='SIFT':
            def SIFTAffnet(img):
                KPlist, sift_des = ComputeSIFTKeypoints(img, Desc = True)
                Identity = np.float32([[1, 0, 0], [0, 1, 0]])
                h, w = img.shape[:2]
                KPlist, sift_des, _ = Filter_Affine_In_Rect(KPlist,Identity,[0,0],[w,h], desc_list = sift_des)
                pyr = buildGaussianPyramid( img, 6 )
                patches, A_list, Ai_list = ComputePatches(KPlist,pyr)
                bP_Alist = getAffmaps_from_Affnet(patches)
                Aq_list = [ ComposeAffineMaps( cv2.invertAffineTransform(bP_Alist[k]), A_list[k] ) for k in range(len(patches)) ]
                return Aq_list
            start_time = time.time()
            Aq_list = SIFTAffnet(img1)
            At_list = SIFTAffnet(img2)
            if dilate:
                dilationSet = [affine_decomp2affine(A) for A in GetCovering(tilts_1_25, endAngle=2*np.pi)]
                if len(Aq_list)<len(At_list):
                    Aq_list = [ComposeAffineMaps(A, B) for A in dilationSet for B in Aq_list]
                else:
                    At_list = [ComposeAffineMaps(A, B) for A in dilationSet for B in At_list]
            ET_Ad = time.time() - start_time
            Aq_list = [affine_decomp(A, ModRots=True) for A in Aq_list]
            At_list = [affine_decomp(A, ModRots=True) for A in At_list]
        
        if AdOPT == 'Greedy':
            FixedTilts = None
            SimuThres = 0.05
        elif AdOPT == 'FixedTilts':
            FixedTilts = GetCovering(tilts_1_7)
            SimuThres = 0.01


        simus1 = SelectSimusFromData( Aq_list, simu_decomp_list=FixedTilts, thres=SimuThres, tilt_radius=np.log(1.7))
        simus2 = SelectSimusFromData( At_list, simu_decomp_list=FixedTilts, thres=SimuThres, tilt_radius=np.log(1.7))
        CreateBaseFig(4)

        CreateBaseFig(1)
        PlotInSpaceOfTilts(Aq_list, density=True)
        DepictDisk(simus1, tilt_radius=np.log(1.7))
        plt.legend(loc='upper left',fontsize=8)
        plt.title('Query')
        plt.savefig('simusquery.png', format='png', dpi=300)
        plt.close(1)

        CreateBaseFig(1)
        PlotInSpaceOfTilts(At_list, density=True)
        DepictDisk(simus2, tilt_radius=np.log(1.7))        
        plt.legend(loc='upper left',fontsize=8)
        plt.title('Target')
        plt.savefig('simustarget.png', format='png', dpi=300)
        plt.close(1)

        t1 = [1.0*A[2] for A in simus1]
        r1 = [1.0*A[3] for A in simus1]
        t2 = [1.0*A[2] for A in simus2]
        r2 = [1.0*A[3] for A in simus2]
        _ = subprocess.call("rm %s/2simu.csv" % dir, shell=True)
        _ = subprocess.call("rm %s/2simu2.csv" % dir, shell=True)
        
        np.savetxt('%s/2simu.csv'% dir, (t1,r1), delimiter=',')
        np.savetxt('%s/2simu2.csv'% dir, (t2,r2), delimiter=',')
        _ = subprocess.check_output("cat %s2simu2.csv >> %s2simu.csv"%(dir,dir), shell=True)       
        _ = subprocess.check_output('cd %s && '% dir +cmdflag+ ' ./imas_bin -im1 "./img1.png" -im2 "./img2.png" -desc %d -covering -0.5 -applyfilter %d > imas.out'%(desc,GFilterCode), shell=True)
    else:
        _ = subprocess.check_output('cd %s && '% dir +cmdflag+ ' ./imas_bin -im1 "./img1.png" -im2 "./img2.png" -desc %d -applyfilter %d -covering %f > imas.out'%(desc,GFilterCode,covering), shell=True)

    df = pandas.read_csv('%sdata_matches.csv'%dir)
    KPlist1 = [cv2.KeyPoint(x = df['x1'][i], y = df[' y1'][i],
            _size = 10, _angle = 0.0,
            _response = 1.0, _octave =  packSIFTOctave(-1,0),
            _class_id = 0) for i in range(df.shape[0])]
    KPlist2 = [cv2.KeyPoint(x = df[' x2'][i], y = df[' y2'][i],
            _size = 10, _angle = 0.0,
            _response = 1.0, _octave =  packSIFTOctave(-1,0),
            _class_id = 0) for i in range(df.shape[0])]
    imas_matches = [cv2.DMatch(i,i,df[' distance'][i]) for i in range(len(KPlist1))]
    imasout = subprocess.check_output('cd %s && cat imas.out' % dir, shell=True).decode('utf-8')
    print(imasout)
    ET_KP = float(subprocess.check_output('cd %s && cat imas.out | grep "IMAS-Detector accomplished in" | cut -f 4 -d" " ' %dir, shell=True).decode('utf-8'))
    ET_M = float(subprocess.check_output('cd %s && cat imas.out | grep "IMAS-Matcher accomplished in" | cut -f 4 -d" " '%dir, shell=True).decode('utf-8'))

    TKPs1 = int(subprocess.check_output('cd %s && cat imas.out | grep "image 1" | cut -f 7 -d" " '%dir, shell=True).decode('utf-8'))
    TKPs2 = int(subprocess.check_output('cd %s && cat imas.out | grep "image 2" | cut -f 7 -d" " '%dir, shell=True).decode('utf-8'))
    Nsimus1 = int(subprocess.check_output('cd %s && cat imas.out | grep "image 1" | cut -f 14 -d" " '%dir, shell=True).decode('utf-8'))
    Nsimus2 = int(subprocess.check_output('cd %s && cat imas.out | grep "image 2" | cut -f 14 -d" " '%dir, shell=True).decode('utf-8'))
    assert len(simus1)+len(simus2)==0 or((AdOPT is not None) or (Nsimus1==len(simus1) and Nsimus2==len(simus2))), repr(AdOPT)+' '+repr(Nsimus1)+' '+repr(len(simus1))+' '+repr(Nsimus2)+' '+repr(len(simus2))

    Ntotal = int(subprocess.check_output('cd %s && cat imas.out | grep "possible matches have been found" | cut -f 4 -d" " '%dir, shell=True).decode('utf-8'))
    if GFilterCode>0:
        Filtered = int(subprocess.check_output('cd %s && cat imas.out | grep "Final number of matches" | cut -f 10 -d" " '%dir, shell=True).decode('utf-8')[:-2])        
        assert Filtered==len(imas_matches)

    if Visual:
        img4 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,imas_matches, None,flags=2)
        cv2.imwrite('./IMASmatches.png',img4)

    return imas_matches


def ASIFTcaller(img1,img2, GFilter='IntCode-2', Visual=False, EuPrecision=24):    
    if GFilter[0:7] == 'IntCode':
        GFilterCode = int(GFilter[8])
    else:
        GFilterCode = 0    

    # example: cmdflag = 'OMP_NUM_THREADS=1'
    cmdflag = ''    

    cv2.imwrite("/tmp/img1.png",img1)
    cv2.imwrite("/tmp/img2.png",img2)
    _ = subprocess.check_output("cp asiftbuild/z_demo_ASIFT /tmp", shell=True)
    _ = subprocess.call("rm /tmp/data_matches.csv", shell=True)
    _ = subprocess.check_output('cd /tmp && '+cmdflag+ ' ./z_demo_ASIFT "./img1.png" "./img2.png" > imas.out', shell=True)

    df = pandas.read_csv('/tmp/data_matches.csv')
    KPlist1 = [cv2.KeyPoint(x = df['x1'][i], y = df[' y1'][i],
            _size = 10, _angle = 0.0,
            _response = 1.0, _octave =  packSIFTOctave(-1,0),
            _class_id = 0) for i in range(df.shape[0])]
    KPlist2 = [cv2.KeyPoint(x = df[' x2'][i], y = df[' y2'][i],
            _size = 10, _angle = 0.0,
            _response = 1.0, _octave =  packSIFTOctave(-1,0),
            _class_id = 0) for i in range(df.shape[0])]
    asift_matches = [cv2.DMatch(i,i,df[' distance'][i]) for i in range(len(KPlist1))]
    imasout = subprocess.check_output('cd /tmp && cat imas.out', shell=True).decode('utf-8')
    print(imasout)
    ET_KP = float(subprocess.check_output('cd /tmp && cat imas.out | grep "Keypoints computation accomplished in" | cut -f 5 -d" " ', shell=True).decode('utf-8'))
    ET_M = float(subprocess.check_output('cd /tmp && cat imas.out | grep "Keypoints matching accomplished in" | cut -f 5 -d" " ', shell=True).decode('utf-8'))

    # TKPs1 = int(subprocess.check_output('cd /tmp && cat imas.out | grep "image 1" | cut -f 7 -d" " ', shell=True).decode('utf-8'))
    # TKPs2 = int(subprocess.check_output('cd /tmp && cat imas.out | grep "image 2" | cut -f 7 -d" " ', shell=True).decode('utf-8'))
    TKPs1 = -1
    TKPs2 = -1
    Nsimus1 = 41
    Nsimus2 = 41

    Ntotal = len(asift_matches)    
    
    if Visual:
        img4 = cv2.drawMatches(img1,KPlist1,img2,KPlist2,asift_matches, None,flags=2)
        cv2.imwrite('./temp/ASIFTmatches.png',img4)        

    return asift_matches
