from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append(".")
import os
import time

from PIL import Image
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import math
import torch.nn.functional as F

from copy import deepcopy

from SparseImgRepresenter import ScaleSpaceAffinePatchExtractor
from LAF import normalizeLAFs, denormalizeLAFs, LAFs2ell, abc2A, convertLAFs_to_A23format
from Utils import line_prepender, batched_forward
from architectures import AffNetFast
from HardNet import HardNet


# USE_CUDA = False
USE_CUDA = torch.cuda.is_available()
WRITE_IMGS_DEBUG = False

AffNetPix = AffNetFast(PS = 32)
weightd_fname = 'hesaffnet/pretrained/AffNet.pth'
if USE_CUDA:
    checkpoint = torch.load(weightd_fname, map_location='cuda:0')
else:
    checkpoint = torch.load(weightd_fname, map_location='cpu')
AffNetPix.load_state_dict(checkpoint['state_dict'])
AffNetPix.eval()


HardNetDescriptor = HardNet()
model_weights = 'hesaffnet/pretrained/HardNet++.pth'
if USE_CUDA:
    hncheckpoint = torch.load(model_weights, map_location='cuda:0')
else:
    hncheckpoint = torch.load(model_weights, map_location='cpu')
HardNetDescriptor.load_state_dict(hncheckpoint['state_dict'])
HardNetDescriptor.eval()

if USE_CUDA:
    AffNetPix = AffNetPix.cuda()
    HardNetDescriptor = HardNetDescriptor.cuda()


from library import *
import cv2

def getAffmaps_from_Affnet(patches_np):
    sp1, sp2 = np.shape(patches_np[0])
    subpatches = torch.autograd.Variable( torch.zeros([len(patches_np), 1, 32, 32], dtype=torch.float32), volatile = True).view(len(patches_np), 1, 32, 32)
    for k in range(0,len(patches_np)):
        subpatch = patches_np[k][int(sp1/2)-16:int(sp2/2)+16, int(sp1/2)-16:int(sp2/2)+16].reshape(1,1,32,32)
        subpatches[k,:,:,:] = torch.from_numpy(subpatch.astype(np.float32)) #=subpatch

    x, y = subpatches.shape[3]/2.0 + 2, subpatches.shape[2]/2.0 + 2     
    LAFs = normalizeLAFs( torch.tensor([[AffNetPix.PS/2, 0, x], [0, AffNetPix.PS/2, y]]).reshape(1,2,3), subpatches.shape[3], subpatches.shape[2] ) 
    baseLAFs = torch.zeros([subpatches.shape[0], 2, 3], dtype=torch.float32) 
    for m in range(subpatches.shape[0]):
       baseLAFs[m,:,:] = LAFs
    
    if USE_CUDA:
        # or ---> A = AffNetPix(subpatches.cuda()).cpu()
        with torch.no_grad():
            A = batched_forward(AffNetPix, subpatches.cuda(), 256).cpu()
    else:
        with torch.no_grad():
            A = AffNetPix(subpatches)
    LAFs = torch.cat([torch.bmm(A,baseLAFs[:,:,0:2]), baseLAFs[:,:,2:] ], dim =2)
    dLAFs = denormalizeLAFs(LAFs, subpatches.shape[3], subpatches.shape[2])
    Alist = convertLAFs_to_A23format( dLAFs.detach().cpu().numpy().astype(np.float32) )
    return Alist


def AffNetHardNet_describe(patches):
    descriptors =  np.zeros( shape = [patches.shape[0], 128], dtype=np.float32)
    HessianAffine = []
    subpatches = torch.autograd.Variable( torch.zeros([len(patches), 1, 32, 32], dtype=torch.float32), volatile = True).view(len(patches), 1, 32, 32)
    baseLAFs = torch.zeros([len(patches), 2, 3], dtype=torch.float32) 
    for m in range(patches.shape[0]):
       patch_np = patches[m,:,:,0].reshape(np.shape(patches)[1:3])
       HessianAffine.append( ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 0, border = 0, num_Baum_iters = 0) )
       with torch.no_grad():
            var_image = torch.autograd.Variable(torch.from_numpy(patch_np.astype(np.float32)), volatile = True)
            patch = var_image.view(1, 1, var_image.size(0),var_image.size(1))
       with torch.no_grad(): 
            HessianAffine[m].createScaleSpace(patch) # to generate scale pyramids and stuff
       x, y = patch.size(3)/2.0 + 2, patch.size(2)/2.0 + 2     
       LAFs = normalizeLAFs( torch.tensor([[AffNetPix.PS/2, 0, x], [0, AffNetPix.PS/2, y]]).reshape(1,2,3), patch.size(3), patch.size(2) ) 
       baseLAFs[m,:,:] = LAFs
       with torch.no_grad():
            subpatch = HessianAffine[m].extract_patches_from_pyr(denormalizeLAFs(LAFs, patch.size(3), patch.size(2)), PS = AffNetPix.PS)
            if WRITE_IMGS_DEBUG:
                SaveImageWithKeys(subpatch.detach().cpu().numpy().reshape([32,32]), [], 'p1/'+str(n)+'.png' )
            # This subpatch has been blured by extract_patches _from_pyr... 
            # let't us crop it manually to obtain fair results agains other methods
            subpatch = patch_np[16:48,16:48].reshape(1,1,32,32)
            #var_image = torch.autograd.Variable(torch.from_numpy(subpatch.astype(np.float32)), volatile = True)
            #subpatch = var_image.view(1, 1, 32,32)
            subpatches[m,:,:,:] = torch.from_numpy(subpatch.astype(np.float32)) #=subpatch
            if WRITE_IMGS_DEBUG:
                SaveImageWithKeys(subpatch.detach().cpu().numpy().reshape([32,32]), [], 'p2/'+str(n)+'.png' )
    if USE_CUDA:
        # or ---> A = AffNetPix(subpatches.cuda()).cpu()
        with torch.no_grad():
            A = batched_forward(AffNetPix, subpatches.cuda(), 256).cpu()
    else:
        with torch.no_grad():
            A = AffNetPix(subpatches)
    LAFs = torch.cat([torch.bmm(A,baseLAFs[:,:,0:2]), baseLAFs[:,:,2:] ], dim =2)
    dLAFs = denormalizeLAFs(LAFs, patch.size(3), patch.size(2))
    Alist = convertLAFs_to_A23format( dLAFs.detach().cpu().numpy().astype(np.float32) )
    for m in range(patches.shape[0]):
       with torch.no_grad():
            patchaff = HessianAffine[m].extract_patches_from_pyr(dLAFs[m,:,:].reshape(1,2,3), PS = 32)
            if WRITE_IMGS_DEBUG:
                SaveImageWithKeys(patchaff.detach().cpu().numpy().reshape([32,32]), [], 'im1/'+str(n)+'.png' )
                SaveImageWithKeys(patch_np, [], 'im2/'+str(n)+'.png' )
            subpatches[m,:,:,:] = patchaff
    if USE_CUDA:
        with torch.no_grad():
            # descriptors = HardNetDescriptor(subpatches.cuda()).detach().cpu().numpy().astype(np.float32)
            descriptors = batched_forward(HardNetDescriptor, subpatches.cuda(), 256).cpu().numpy().astype(np.float32)
    else:
        with torch.no_grad():
            descriptors = HardNetDescriptor(subpatches).detach().cpu().numpy().astype(np.float32)
    return descriptors, Alist

def AffNetHardNet_describeFromKeys(img_np, KPlist):
    img = torch.autograd.Variable(torch.from_numpy(img_np.astype(np.float32)), volatile = True)
    img = img.view(1, 1, img.size(0),img.size(1))
    HessianAffine = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = 0, border = 0, num_Baum_iters = 0)
    if USE_CUDA:
        HessianAffine = HessianAffine.cuda()
        img = img.cuda()
    with torch.no_grad():
        HessianAffine.createScaleSpace(img) # to generate scale pyramids and stuff
    descriptors = []
    Alist = []
    n=0
    # for patch_np in patches:
    for kp in KPlist: 
       x, y = np.float32(kp.pt)
       LAFs = normalizeLAFs( torch.tensor([[AffNetPix.PS/2, 0, x], [0, AffNetPix.PS/2, y]]).reshape(1,2,3), img.size(3), img.size(2) )
       with torch.no_grad():
            patch = HessianAffine.extract_patches_from_pyr(denormalizeLAFs(LAFs, img.size(3), img.size(2)), PS = AffNetPix.PS)
       if WRITE_IMGS_DEBUG:
            SaveImageWithKeys(patch.detach().cpu().numpy().reshape([32,32]), [], 'p2/'+str(n)+'.png' )
       if USE_CUDA:
            # or ---> A = AffNetPix(subpatches.cuda()).cpu()
            with torch.no_grad():
                A = batched_forward(AffNetPix, patch.cuda(), 256).cpu()
       else:
            with torch.no_grad():
                A = AffNetPix(patch)
       new_LAFs = torch.cat([torch.bmm(A,LAFs[:,:,0:2]), LAFs[:,:,2:] ], dim =2)
       dLAFs = denormalizeLAFs(new_LAFs, img.size(3), img.size(2))
       with torch.no_grad():
            patchaff = HessianAffine.extract_patches_from_pyr(dLAFs, PS = 32)
            if WRITE_IMGS_DEBUG:
                SaveImageWithKeys(patchaff.detach().cpu().numpy().reshape([32,32]), [], 'p1/'+str(n)+'.png' )
                SaveImageWithKeys(img_np, [kp], 'im1/'+str(n)+'.png' )
            descriptors.append( HardNetDescriptor(patchaff).cpu().numpy().astype(np.float32) )
            Alist.append( convertLAFs_to_A23format( LAFs.detach().cpu().numpy().astype(np.float32) ) )
    n=n+1
    return descriptors, Alist


def HessAffNetHardNet_Detect(img, Nfeatures=500):
    var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)), volatile = True)
    var_image_reshape = var_image.view(1, 1, var_image.size(0),var_image.size(1))
    HessianAffine = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = Nfeatures, border = 5, num_Baum_iters = 1,  AffNet = AffNetPix)
    if USE_CUDA:
        HessianAffine = HessianAffine.cuda()
        var_image_reshape = var_image_reshape.cuda()
        
    with torch.no_grad():
        LAFs, responses = HessianAffine(var_image_reshape, do_ori = True)

    # these are my affine maps to work with
    Alist = convertLAFs_to_A23format(LAFs).cpu().numpy().astype(np.float32)
    KPlist = [cv2.KeyPoint(x=A[0,2], y=A[1,2], _size=10, _angle=0.0,
                               _response=1, _octave=packSIFTOctave(0,0),_class_id=1)
                                for A in Alist]
    return KPlist, Alist, responses


def HessAffNetHardNet_DetectAndDescribe(img, Nfeatures=500):
    var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)), volatile = True)
    var_image_reshape = var_image.view(1, 1, var_image.size(0),var_image.size(1))
    HessianAffine = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = Nfeatures, border = 5, num_Baum_iters = 1,  AffNet = AffNetPix)
    if USE_CUDA:
        HessianAffine = HessianAffine.cuda()
        var_image_reshape = var_image_reshape.cuda()
        
    with torch.no_grad():
        LAFs, responses = HessianAffine(var_image_reshape, do_ori = True)
        patches = HessianAffine.extract_patches_from_pyr(LAFs, PS = 32)
        descriptors = HardNetDescriptor(patches)

    # these are my affine maps to work with
    Alist = convertLAFs_to_A23format(LAFs).cpu().numpy().astype(np.float32)
    KPlist = [cv2.KeyPoint(x=A[0,2], y=A[1,2], _size=10, _angle=0.0,
                               _response=1, _octave=packSIFTOctave(0,0),_class_id=1)
                                for A in Alist]
    return KPlist, patches, descriptors, Alist, responses

def HessAff_Detect(img, PatchSize=60, Nfeatures=500):
    var_image = torch.autograd.Variable(torch.from_numpy(img.astype(np.float32)), volatile = True)
    var_image_reshape = var_image.view(1, 1, var_image.size(0),var_image.size(1))
    HessianAffine = ScaleSpaceAffinePatchExtractor( mrSize = 5.192, num_features = Nfeatures, border = PatchSize/2, num_Baum_iters = 1)
    # if USE_CUDA:
    #     HessianAffine = HessianAffine.cuda()
    #     var_image_reshape = var_image_reshape.cuda()
        
    with torch.no_grad():
        LAFs, responses = HessianAffine(var_image_reshape, do_ori = True)
        patches = HessianAffine.extract_patches_from_pyr(LAFs, PS = PatchSize).cpu()

    # these are my affine maps to work with
    Alist = convertLAFs_to_A23format(LAFs).cpu().numpy().astype(np.float32)
    KPlist = [cv2.KeyPoint(x=A[0,2], y=A[1,2], _size=10, _angle=0.0,
                               _response=1, _octave=packSIFTOctave(0,0),_class_id=1)
                                for A in Alist]
    return KPlist, np.array(patches), Alist, responses.cpu()

from Losses import distance_matrix_vector
def BruteForce4HardNet(descriptors1, descriptors2, SNN_threshold = 0.8):
    if type(descriptors1)!=torch.Tensor:
        descriptors1 = torch.from_numpy(descriptors1.astype(np.float32))
        descriptors2 = torch.from_numpy(descriptors2.astype(np.float32))
    # if USE_CUDA:
    #     descriptors1 = descriptors1.cuda()
    #     descriptors2 = descriptors2.cuda()
    #Bruteforce matching with SNN threshold    
    dist_matrix = distance_matrix_vector(descriptors1, descriptors2)
    min_dist, idxs_in_2 = torch.min(dist_matrix,1)
    dist_matrix[:,idxs_in_2] = 100000;# mask out nearest neighbour to find second nearest
    min_2nd_dist, idxs_2nd_in_2 = torch.min(dist_matrix,1)
    mask = (min_dist / (min_2nd_dist + 1e-8)) <= SNN_threshold

    tent_matches_in_1 = indxs_in1 = torch.autograd.Variable(torch.arange(0, idxs_in_2.size(0)), requires_grad = False)[mask]
    tent_matches_in_2 = idxs_in_2[mask]

    tent_matches_in_1 = tent_matches_in_1.data.cpu().long()
    tent_matches_in_2 = tent_matches_in_2.data.cpu().long()
    return tent_matches_in_1, tent_matches_in_2
