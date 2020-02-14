import numpy as np
import cv2
import math
import time
import glob, os
import random
import psutil
import ctypes
from datetime import datetime

MaxSameKP_dist = 5 # pixels
MaxSameKP_angle = 10 # degrees

class ClassSIFTparams():
    def __init__(self, nfeatures = 0, nOctaveLayers = 3, contrastThreshold = 0.04, edgeThreshold = 10, sigma = 1.6, firstOctave = -1, sift_init_sigma = 0.5, graydesc = True):
        self.nOctaves = 4
        self.nfeatures = nfeatures
        self.nOctaveLayers = nOctaveLayers
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.sigma = sigma
        self.firstOctave = firstOctave
        self.sift_init_sigma = sift_init_sigma
        self.graydesc = graydesc
        self.flt_epsilon = 1.19209e-07
        self.lambda_descr = 6
        self.new_radius_descr = 29.5


siftparams = ClassSIFTparams(graydesc = True)



def SimulateAffineMap(zoom_step,psi,t1_step,phi,img0,mask=None, CenteredAt=None, t2_step = 1.0, inter_flag = cv2.INTER_CUBIC, border_flag = cv2.BORDER_CONSTANT, SimuBlur = True):
    '''
    Computing affine deformations of images as in [https://rdguez-mariano.github.io/pages/imas]
    Let A = R_psi0 * diag(t1,t2) * R_phi0    with t1>t2
          = lambda * R_psi0 * diag(t1/t2,1) * R_phi0

    Parameters given should be as:
    zoom_step = 1/lambda
    t1_step = 1/t1
    t2_step = 1/t2
    psi = -psi0 (in degrees)
    phi = -phi0 (in degrees)

    ASIFT proposed params:
    inter_flag = cv2.INTER_LINEAR
    SimuBlur = True

    Also, another kind of exterior could be:
    border_flag = cv2.BORDER_REPLICATE
    '''

    tx = zoom_step*t1_step
    ty = zoom_step*t2_step
    assert tx>=1 and ty>=1, 'Either scale or t are defining a zoom-in operation. If you want to zoom-in do it manually. tx = '+str(tx)+', ty = '+str(ty)

    img = img0.copy()
    arr = []
    DoCenter = False
    if type(CenteredAt) is list:
        DoCenter = True
        arr = np.array(CenteredAt).reshape(-1,2)

    h, w = img.shape[:2]
    tcorners = SquareOrderedPts(h,w,CV=False)
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A1 = np.float32([[1, 0, 0], [0, 1, 0]])

    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A1 = np.float32([[c,-s], [ s, c]])
        tcorners = np.dot(tcorners, A1.T)
        x, y, w, h = cv2.boundingRect(np.int32(tcorners).reshape(1,-1,2))
        A1 = np.hstack([A1, [[-x], [-y]]])
        if DoCenter and tx == 1.0 and ty == 1.0 and psi == 0.0:
            arr = AffineArrayCoor(arr,A1)[0].ravel()
            h0, w0 = img0.shape[:2]
            A1[0][2] += h0/2.0 - arr[0]
            A1[1][2] += w0/2.0 - arr[1]
            w, h = w0, h0
            img = cv2.warpAffine(img, A1, (w, h), flags=inter_flag, borderMode=border_flag)
        else:
            img = cv2.warpAffine(img, A1, (w, h), flags=inter_flag, borderMode=border_flag)


    h, w = img.shape[:2]
    A2 = np.float32([[1, 0, 0], [0, 1, 0]])
    tcorners = SquareOrderedPts(h,w,CV=False)
    if tx != 1.0 or ty != 1.0:
        sx = 0.8*np.sqrt(tx*tx-1)
        sy = 0.8*np.sqrt(ty*ty-1)
        if SimuBlur:
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=sx, sigmaY=sy)
        A2[0] /= tx
        A2[1] /= ty

    if psi != 0.0:
        psi = np.deg2rad(psi)
        s, c = np.sin(psi), np.cos(psi)
        Apsi = np.float32([[c,-s], [ s, c]])
        Apsi = np.matmul(Apsi,A2[0:2,0:2])
        tcorners = np.dot(tcorners, Apsi.T)
        x, y, w, h = cv2.boundingRect(np.int32(tcorners).reshape(1,-1,2))
        A2[0:2,0:2] = Apsi
        A2[0][2] -= x
        A2[1][2] -= y



    if tx != 1.0 or ty != 1.0 or psi != 0.0:
        if DoCenter:
            A = ComposeAffineMaps(A2,A1)
            arr = AffineArrayCoor(arr,A)[0].ravel()
            h0, w0 = img0.shape[:2]
            A2[0][2] += h0/2.0 - arr[0]
            A2[1][2] += w0/2.0 - arr[1]
            w, h = w0, h0
        img = cv2.warpAffine(img, A2, (w, h), flags=inter_flag, borderMode=border_flag)

    A = ComposeAffineMaps(A2,A1)

    if psi!=0 or phi != 0.0 or tx != 1.0 or ty != 1.0:
        if DoCenter:
            h, w = img0.shape[:2]
        else:
            h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=inter_flag)
    Ai = cv2.invertAffineTransform(A)
    return img, mask, A, Ai


def unpackSIFTOctave(kp, XI=False):
    ''' Opencv packs the true octave, scale and layer inside kp.octave.
    This function computes the unpacking of that information.
    '''
    _octave = kp.octave
    octave = _octave&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1/(1<<octave))
    else:
        scale = float(1<<-octave)

    if XI:
        yi = (_octave>>16)&0xFF
        xi = yi/255.0 - 0.5
        return octave, layer, scale, xi
    else:
        return octave, layer, scale

def packSIFTOctave(octave, layer, xi=0.0):
    po = octave&0xFF
    pl = (layer&0xFF)<<8
    pxi = round((xi + 0.5)*255.0)&0xFF
    pxi = pxi<<16
    return  po + pl + pxi

def DescRadius(kp, InPyr=False, SIFT=False):
    ''' Computes the Descriptor radius with respect to either an image
        in the pyramid or to the original image.
    '''
    factor = siftparams.new_radius_descr
    if SIFT:
        factor = siftparams.lambda_descr
    if InPyr:
        o, l, s = unpackSIFTOctave(kp)
        return( np.float32(kp.size*s*factor*0.5) )
    else:
        return( np.float32(kp.size*factor*0.5) )


def AngleDiff(a,b, InRad=False):
    ''' Computes the Angle Difference between a and b.
        0<=a,b<=360
    '''
    if InRad:
        a = np.rad2deg(a) % 360
        b = np.rad2deg(b) % 360
    assert a>=0 and a<=360 and b>=0 and b<=360, 'a = '+str(a)+', b = '+str(b)
    anglediff = abs(a-b)% 360
    if anglediff > 180:
        anglediff = 360 - anglediff
    
    if InRad:
        return np.deg2rad(anglediff)
    else:
        return anglediff


def features_deepcopy(f):
    return [cv2.KeyPoint(x = k.pt[0], y = k.pt[1],
            _size = k.size, _angle = k.angle,
            _response = k.response, _octave = k.octave,
            _class_id = k.class_id) for k in f]

def matches_deepcopy(f):
    return [cv2.DMatch(_queryIdx=k.queryIdx, _trainIdx=k.trainIdx, _distance=k.distance) for k in f]


def Filter_Affine_In_Rect(kp_list, A, p_min, p_max, desc_list = None, isSIFT=False):
    ''' Filters out all descriptors in kp_list that do not lay inside the
    the parallelogram defined by the image of a rectangle by the affine transform A.
    The rectangle is defined by (p_min,p_max).
    '''
    desc_listing = False
    desc_list_in = []
    desc_pos = 0
    if type(desc_list) is np.ndarray:
        desc_listing = True
        desc_list_in = desc_list.copy()
    x1, y1 = p_min[:2]
    x2, y2 = p_max[:2]
    Ai = cv2.invertAffineTransform(A)
    kp_back = AffineKPcoor(kp_list,Ai)
    kp_list_in = []
    kp_list_out = []
    cyclic_corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
    cyclic_corners = AffineArrayCoor(cyclic_corners,A)
    for i in range(0,np.size(kp_back)):
        if kp_back[i].pt[0]>=x1 and kp_back[i].pt[0]<x2 and kp_back[i].pt[1]>=y1 and kp_back[i].pt[1]<y2:
            In = True
            r = DescRadius(kp_list[i],SIFT=isSIFT)*1.4142
            for j in range(0,4):
                if r > dist_pt_to_line(kp_list[i].pt,cyclic_corners[j],cyclic_corners[j+1]):
                    In = False
            if In == True:
                if desc_listing:
                    desc_list_in[desc_pos,:]= desc_list[i,:]
                    desc_pos +=1
                kp_list_in.append(kp_list[i])
            else:
                kp_list_out.append(kp_list[i])
        else:
            kp_list_out.append(kp_list[i])
    if desc_listing:
        return kp_list_in, desc_list_in[:desc_pos,:], kp_list_out
    else:
        return kp_list_in, kp_list_out



def dist_pt_to_line(p,p1,p2):
    ''' Computes the distance of a point (p) to a line defined by two points (p1, p2). '''
    x0, y0 = np.float32(p[:2])
    x1, y1 = np.float32(p1[:2])
    x2, y2 = np.float32(p2[:2])
    dist = abs( (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1 ) / np.sqrt( pow(y2-y1,2) + pow(x2-x1,2) )
    return dist


def PolarCoor_from_vector(p_source,p_arrow):
    ''' It computes the \rho and \theta such that
        \rho * exp( i * \theta ) = p_arrow-p_source
    '''
    p = np.array(p_arrow)- np.array(p_source)
    rho = np.linalg.norm(p)
    theta = 0
    if rho>0:
        theta = np.arctan2(p[1],p[0])
        theta = np.rad2deg(theta % (2 * np.pi))
    return  rho, theta


def ComposeAffineMaps(A_lhs,A_rhs):
    ''' Comutes the composition of affine maps:
        A = A_lhs ∘ A_rhs
    '''
    A = np.matmul(A_lhs[0:2,0:2],A_rhs)
    A[:,2] += A_lhs[:,2]
    return A

def kp2LocalAffine(kp, w=60,h=60):
    ''' Computes the affine map A such that: for any x 
    living in the image coordinates A(x) is the 
    corresponding coordinates of x in the patch computed 
    from the keypoint kp.
    '''
    scale = siftparams.new_radius_descr/DescRadius(kp)
    x, y= kp.pt[0], kp.pt[1]
    angle = 360.0 - kp.angle
    if(np.abs(angle - 360.0) < siftparams.flt_epsilon):
        angle = 0.0
    phi = np.deg2rad(angle)
    s, c = np.sin(phi), np.cos(phi)
    R = np.float32([[c,-s], [ s, c]])
    A = scale * np.float32([[1, 0, -x], [0, 1, -y]])
    A = np.matmul(R,A)
    A[:,2] += np.array([w/2, h/2])
    return A    


def AffineArrayCoor(arr,A):
    if type(arr) is list:
        arr = np.array(arr).reshape(-1,2)
    AA = A[0:2,0:2]
    Ab = A[:,2]
    arr_out = []
    for j in range(0,arr.shape[0]):
        arr_out.append(np.matmul(AA,np.array(arr[j,:])) + Ab )
    return np.array(arr_out)

def AffineKPcoor(kp_list,A, Pt_mod = True, Angle_mod = True):
    ''' Transforms information details on each kp_list keypoints by following
        the affine map A.
    '''
    kp_affine = features_deepcopy(kp_list)
    AA = A[0:2,0:2]
    Ab = A[:,2]
    for j in range(0,np.size(kp_affine)):
        newpt = tuple( np.matmul(AA,np.array(kp_list[j].pt)) + Ab)
        if Pt_mod:
            kp_affine[j].pt = newpt
        if Angle_mod:
            phi = np.deg2rad(kp_list[j].angle)
            s, c = np.sin(phi), np.cos(phi)
            R = np.float32([[c,-s], [ s, c]])
            p_arrow = np.matmul( R , [50.0, 0.0] ) + np.array(kp_list[j].pt)
            p_arrow = tuple( np.matmul(AA,p_arrow) + Ab)
            rho, kp_affine[j].angle =  PolarCoor_from_vector(newpt, p_arrow)
    return kp_affine


def affine_decomp2affine(vec):
    lambda_scale = vec[0]
    phi2 = vec[1]
    t = vec[2]
    phi1 = vec[3]

    s, c = np.sin(phi1), np.cos(phi1)
    R_phi1 = np.float32([[c,s], [ -s, c]])
    s, c = np.sin(phi2), np.cos(phi2)
    R_phi2 = np.float32([[c,s], [ -s, c]])

    A = lambda_scale * np.matmul(R_phi2, np.matmul(np.diag([t,1.0]),R_phi1) )
    if np.shape(vec)[0]==6:
        A = np.concatenate(( A, [[vec[4]], [vec[5]]] ), axis=1)
    return A


def affine_decomp(A0,doAssert=True, ModRots=False):
    '''Decomposition of a 2x2 matrix A (whose det(A)>0) satisfying
        A = lambda*R_phi2*diag(t,1)*R_phi1.
        where lambda and t are scalars, and R_phi1, R_phi2 are rotations.
    '''
    epsilon = 0.0001
    A = A0[0:2,0:2]
    Adet = np.linalg.det(A)
    if doAssert:
        assert Adet>0

    if Adet>0:
        #   A = U * np.diag(s) * V
        U, s, V = np.linalg.svd(A, full_matrices=True)
        T = np.diag(s)
        K = np.float32([[-1, 0], [0, 1]])

        # K*D*K = D
        if ((np.linalg.norm(np.linalg.det(U)+1)<=epsilon) and (np.linalg.norm(np.linalg.det(V)+1)<=epsilon)):
            U = np.matmul(U,K)
            V = np.matmul(K,V)

        phi2_drift = 0.0
        # Computing First Rotation
        phi1 = np.arctan2( V[0,1], V[0,0] )
        if ModRots and phi1<0:
            phi1 = phi1 + np.pi
            phi2_drift = -np.pi

        # Computing Second Rotation
        phi2 = np.mod(  np.arctan2( U[0,1],U[0,0]) + phi2_drift  , 2.0*np.pi)

        # Computing Tilt and Lambda
        lambda_scale = T[1,1]
        T[0,0]=T[0,0]/T[1,1]
        T[1,1]=1.0

        if T[0,0]-1.0<=epsilon:
            phi2 = np.mod(phi1+phi2,2.0*np.pi)
            phi1 = 0.0
        
        s, c = np.sin(phi1), np.cos(phi1)
        R_phi1 = np.float32([[c,s], [ -s, c]])
        s, c = np.sin(phi2), np.cos(phi2)
        R_phi2 = np.float32([[c,s], [ -s, c]])
        
        temp = lambda_scale*np.matmul( R_phi2 ,np.matmul(T,R_phi1) )

        # Couldnt decompose A
        if doAssert and np.linalg.norm(A - temp,'fro')>epsilon:
            print('Error: affine_decomp couldnt really decompose A')
            print(A0)
            print('----- end of A')

        rvec = [lambda_scale, phi2, T[0,0], phi1]
        if np.shape(A0)[1]==3:
            rvec = np.concatenate(( rvec, [A0[0,2], A0[1,2]] ))
    else:
        rvec = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    return rvec


def transition_tilt( Avec, Bvec ):
    ''' Computes the transition tilt between two affine maps as in [https://rdguez-mariano.github.io/pages/imas]
    Let
    A = lambda1 * R_phi1 * diag(t,1) * psi1
    B = lambda2 * R_phi2 * diag(s,1) * psi2
    then Avec and Bvec are respectively the affine_decomp of A and B
    '''
    t = Avec[2]
    psi1 = Avec[3]
    s = Bvec[2]
    psi2 = Bvec[3]
    cos_2 = pow( np.cos(psi1-psi2), 2.0)
    g = ( pow(t/s, 2.0) + 1.0 )*cos_2 + ( 1.0/pow(s, 2.0) + pow(t,2.0) )*( 1.0 - cos_2 )
    G = (s/t)*g/2.0
    tau = G + np.sqrt( pow(G,2.0) - 1.0 )
    return tau


tilts_1_25 = [1.0, np.pi, 1.34982, 0.790026, 1.69436, 0.395013] # covers a region=1.71 with disks radius of 1.25
tilts_1_15 = [1.0, np.pi, 1.26938, 0.524565, 1.43195, 0.349323, 1.68408, 0.243276]  # covers a region=1.708 disks with radius of 1.15
tilts_1_7 = [1.0, np.pi,  2.6175699234, 0.3980320096, 5.1816802025, 0.1983139962]
tilts_1_4 = [1.0, np.pi, 1.7745000124, 0.5254489779, 2.3880701065, 0.2863940001, 3.9156799316, 0.1429599971]
tilts_2 = [1.0, np.pi, 3.33005, 0.455102]
def GetCovering(coveringcode, endAngle=np.pi):
    covering=tuple([])
    for i in range(int(len(coveringcode)/2)):
        covering = covering + tuple([[1.0, 0.0, coveringcode[2*i], x, 0.0, 0.0] for x in np.arange(0.0, endAngle, coveringcode[2*i+1])])
    return covering

def SelectSimusFromData(A_decomp_list, simu_decomp_list=None, tilt_radius=np.log(1.7), thres=-1.0, Depict=True ):
    res = []
    if simu_decomp_list is None:
        Adecomp_list = np.array(A_decomp_list).copy()
        while len(Adecomp_list)>0 and len(Adecomp_list)>len(A_decomp_list)*thres:        
            BestScore = 0
            BestAnnears, BestA = [], []
            for A in Adecomp_list:
                Annears = np.array([transition_tilt(A,B) for B in Adecomp_list])>np.exp(tilt_radius)
                Ascore = len(Annears) - np.sum(Annears)
                if Ascore>BestScore:
                    BestA = A
                    BestScore = Ascore
                    BestAnnears = Annears
            Adecomp_list = Adecomp_list[BestAnnears]
            res.append(BestA)
    else:
        for s in simu_decomp_list:
            count = 0
            for a in A_decomp_list:
                if transition_tilt(s, a)<np.exp(tilt_radius):
                    count=count+1
            if float(count/len(A_decomp_list))>thres:
                res.append(s)            
    return res


def ComputeSIFTKeypoints(img, Desc = False, MaxNum = -1):
    gray = []
    if len(img.shape)!=2:
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = img.view()

    sift = cv2.xfeatures2d.SIFT_create(
    nfeatures = siftparams.nfeatures,
    nOctaveLayers = siftparams.nOctaveLayers, contrastThreshold = siftparams.contrastThreshold,
    edgeThreshold = siftparams.edgeThreshold, sigma = siftparams.sigma
    )
    if Desc:
        kp, des = sift.detectAndCompute(gray,None)
        if MaxNum>0 and len(kp)>MaxNum:
            responses = [k.response for k in kp]
            idxs = np.fliplr( np.reshape(np.argsort(responses),(1,-1)) ).reshape(-1) 
            kpF = []
            desF =np.zeros(shape=(MaxNum,des.shape[1]), dtype=des.dtype)
            for n in range(MaxNum):
                kpF.append( kp[idxs[n]] )
                desF[n,:] = des[idxs[n],:]
            return kpF, desF
        else:
            return kp, des
    else:
        kp = sift.detect(gray,None)
        if MaxNum>0 and len(kp)>MaxNum:
            responses = [k.response for k in kp]
            idxs = np.fliplr( np.reshape(np.argsort(responses),(1,-1)) ).reshape(-1) 
            kpF = []
            for n in range(MaxNum):
                kpF.append( kp[idxs[n]] )
            return kpF
        else:            
            return kp


def ComputePatches(kp_list,gpyr, border_mode = cv2.BORDER_CONSTANT):
    ''' Computes the associated patch to each keypoint in kp_list.
        Returns:
        img_list - list of patches.
        A_list - lists of affine maps A such that A(BackgroundImage)*1_{[0,2r]x[0,2r]} = patch.
        Ai_list - list of the inverse of the above affine maps.

    '''
    img_list = []
    A_list = []
    Ai_list = []
    for i in range(0,np.size(kp_list)):
        kpt = kp_list[i]
        octave, layer, scale = unpackSIFTOctave(kpt)
        assert octave >= siftparams.firstOctave and layer <= siftparams.nOctaveLayers+2, 'octave = '+str(octave)+', layer = '+str(layer)
        # formula in opencv:  kpt.size = sigma*powf(2.f, (layer + xi) / nOctaveLayers)*(1 << octv)*2
        step = kpt.size*scale*0.5 # sigma*powf(2.f, (layer + xi) / nOctaveLayers)
        ptf = np.array(kpt.pt)*scale
        angle = 360.0 - kpt.angle
        if(np.abs(angle - 360.0) < siftparams.flt_epsilon):
            angle = 0.0

        img = gpyr[(octave - siftparams.firstOctave)*(siftparams.nOctaveLayers + 3) + layer]

        r = siftparams.new_radius_descr

        phi = np.deg2rad(angle)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]]) / step
        Rptf = np.matmul(A,ptf)
        x = Rptf[0]-r
        y = Rptf[1]-r
        A = np.hstack([A, [[-x], [-y]]])

        dim = np.int32(2*r+1)
        img = cv2.warpAffine(img, A, (dim, dim), flags=cv2.INTER_CUBIC, borderMode=border_mode)
        #print('Octave =', octave,'; Layer =', layer, '; Scale =', scale,'; Angle =',angle)

        oA = np.float32([[1, 0, 0], [0, 1, 0]]) * scale
        A = ComposeAffineMaps(A,oA)
        Ai = cv2.invertAffineTransform(A)
        img_list.append(img.astype(np.float32))
        A_list.append(A)
        Ai_list.append(Ai)
    return img_list, A_list, Ai_list


def ComputeSimilaritiesFromKPs(kp_list):
    A_list = []
    Ai_list = []
    for i in range(0,np.size(kp_list)):
        kpt = kp_list[i]
        octave, layer, scale = unpackSIFTOctave(kpt)
        assert octave >= siftparams.firstOctave and layer <= siftparams.nOctaveLayers+2, 'octave = '+str(octave)+', layer = '+str(layer)
        step = kpt.size*scale*0.5 # sigma*powf(2.f, (layer + xi) / nOctaveLayers)
        ptf = np.array(kpt.pt)*scale
        angle = 360.0 - kpt.angle
        if(np.abs(angle - 360.0) < siftparams.flt_epsilon):
            angle = 0.0

        r = siftparams.new_radius_descr

        phi = np.deg2rad(angle)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]]) / step
        Rptf = np.matmul(A,ptf)
        x = Rptf[0]-r
        y = Rptf[1]-r
        A = np.hstack([A, [[-x], [-y]]])

        oA = np.float32([[1, 0, 0], [0, 1, 0]]) * scale
        A = ComposeAffineMaps(A,oA)
        Ai = cv2.invertAffineTransform(A)
        A_list.append(A)
        Ai_list.append(Ai)
    return A_list, Ai_list

def SaveImageWithKeys(img, keys, pathname, rootfolder='temp/', Flag=2):
        colors=( (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255) )
        if len(np.shape(img))==2:
            patch = cv2.cvtColor( img.astype(np.uint8), cv2.COLOR_GRAY2RGB )
        else:
            patch = img.astype(np.uint8)
        if len(keys)>len(colors):
            patch=cv2.drawKeypoints(patch,keys,patch, flags=Flag)
        else:
            for n in range(np.min([len(colors),len(keys)])):
                patch=cv2.drawKeypoints(patch,keys[n:n+1],patch, color=colors[n] ,flags=Flag)
        cv2.imwrite(rootfolder+pathname,patch)


def buildGaussianPyramid( base, LastOctave ):
    '''
    Computing the Gaussian Pyramid as in opencv SIFT
    '''
    if siftparams.graydesc and len(base.shape)!=2:
        base = cv2.cvtColor(base,cv2.COLOR_BGR2GRAY)
    else:
        base = base.copy()

    if siftparams.firstOctave<0:
        sig_diff = np.sqrt( max(siftparams.sigma * siftparams.sigma - siftparams.sift_init_sigma * siftparams.sift_init_sigma * 4, 0.01) )
        base = cv2.resize(base, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR_EXACT)

    rows, cols = base.shape[:2]

    nOctaves = np.round(np.log( np.float32(min( cols, rows ))) / np.log(2.0) - 2) - siftparams.firstOctave
    nOctaves = min(nOctaves,LastOctave)
    nOctaves = np.int32(nOctaves)

    sig = ([siftparams.sigma])
    k = np.float32(pow( 2.0 , 1.0 / np.float32(siftparams.nOctaveLayers) ))

    for i in range(1,siftparams.nOctaveLayers + 3):
        sig_prev = pow(k, np.float32(i-1)) * siftparams.sigma
        sig_total = sig_prev*k
        sig += ([ np.sqrt(sig_total*sig_total - sig_prev*sig_prev) ])

    assert np.size(sig) == siftparams.nOctaveLayers + 3

    pyr = []
    for o in range(nOctaves):
        for i in range(siftparams.nOctaveLayers + 3):
            pyr.append([])

    assert len(pyr) == nOctaves*(siftparams.nOctaveLayers + 3)


    for o in range(nOctaves):
        for i in range(siftparams.nOctaveLayers + 3):
            if o == 0  and  i == 0:
                pyr[o*(siftparams.nOctaveLayers + 3) + i] = base.copy()
            elif i == 0:
                src = pyr[(o-1)*(siftparams.nOctaveLayers + 3) + siftparams.nOctaveLayers]
                srcrows, srccols = src.shape[:2]
                pyr[o*(siftparams.nOctaveLayers + 3) + i] = cv2.resize(src, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            else:
                src = pyr[o*(siftparams.nOctaveLayers + 3) + i-1]
                pyr[o*(siftparams.nOctaveLayers + 3) + i] = cv2.GaussianBlur(src, (0, 0), sigmaX=sig[i], sigmaY=sig[i])
    return(pyr)

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

def FirstOrderApprox_Homography(H0, X0=np.array([[0],[0],[1]])):
    ''' Computes the first order Taylor approximation (which is an affine map)
    of the Homography H0 centered at X0 (X0 is in homogeneous coordinates).
    '''
    X0 = np.array( X0 ).reshape(-1,1)
    H = H0.copy()
    col3 = np.matmul(H,X0)
    H[:,2] = col3.reshape(3)
    A = np.zeros((2,3), dtype = np.float32)
    A[0:2,0:2] = H[0:2,0:2]/H[2,2] - np.array([ H[0,2]*H[2,0:2], H[1,2]*H[2,0:2] ])/pow(H[2,2],2)
    A[:,2] = H[0:2,2]/H[2,2] - np.matmul( A[0:2,0:2], X0[0:2,0]/X0[2,0] )
    return A


def AffineFit(Xi,Yi):
    assert np.shape(Xi)[0]==np.shape(Yi)[0] and np.shape(Xi)[1]==2 and np.shape(Yi)[1]==2
    n = np.shape(Xi)[0]
    A = np.zeros((2*n,6),dtype=np.float32)
    b = np.zeros((2*n,1),dtype=np.float32)
    for i in range(0,n):
        A[2*i,0] = Xi[i,0]
        A[2*i,1] = Xi[i,1]
        A[2*i,2] = 1.0
        A[2*i+1,3] = Xi[i,0]
        A[2*i+1,4] = Xi[i,1]
        A[2*i+1,5] = 1.0

        b[2*i,0] = Yi[i,0]
        b[2*i+1,0] = Yi[i,1]
    result = np.linalg.lstsq(A,b,rcond=None)
    return result[0].reshape((2, 3))


def SquareOrderedPts(hs,ws,CV=True):
    # Patch starts from the origin
    ws = ws - 1
    hs = hs - 1
    if CV:
        return [
            cv2.KeyPoint(x = 0,  y =0, _size = 10, _angle = 0, _response = 1.0, _octave = 0, _class_id = 0),
            cv2.KeyPoint(x = ws, y =0, _size = 10, _angle = 0, _response = 1.0, _octave = 0, _class_id = 0),
            cv2.KeyPoint(x = ws, y =hs, _size = 10, _angle = 0, _response = 1.0, _octave = 0, _class_id = 0),
            cv2.KeyPoint(x = 0,  y =hs, _size = 10, _angle = 0, _response = 1.0, _octave = 0, _class_id = 0)
            ]
    else:
        # return np.float32([ [0,0], [ws+1,0], [ws+1, hs+1], [0,hs+1] ])
        return np.float32([ [0,0], [ws,0], [ws, hs], [0,hs] ])

def Flatten2Pts(vec):
    X = np.zeros( (np.int32(len(vec)/2), 2), np.float32)
    X[:,0] = vec[0::2]
    X[:,1] = vec[1::2]
    return X

def Pts2Flatten(X):
    h,w= np.shape(X)[:2]
    vec = np.zeros( (h*w), np.float32)
    vec[0::2] = X[:,0]
    vec[1::2] = X[:,1]
    return vec

def close_per(vec):
    return( np.array(tuple(vec)+tuple([vec[0]])) )

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def HumanElapsedTime(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    # print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    return hours, minutes, seconds