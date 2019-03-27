#!/usr/bin/python

import os
import sys
import argparse
import random
import quippy as qp
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import scale
from sklearn import svm

def correlation_factor(PCAFiles, al, Z, nPCA):
    structIdxs = []
    for i, at in enumerate(al):
        n = 0
        atoms = at.get_atomic_numbers()
        for j in Z:
            n += np.count_nonzero(atoms == j)
        structIdxs.append(n)
    
    f = 0
    s = 0
    ss = 0
    CF = []
    if os.path.splitext(PCAFiles[f])[1] == '.npy':
        batchPCA = np.load(PCAFiles[f])[:, 0:nPCA]
    else:
        batchPCA = np.loadtxt(PCAFiles[f])[:, 0:nPCA]

    for i in range(0, len(structIdxs)):
        iPCA = batchPCA[s:s+structIdxs[i]]
        iMean = np.mean(iPCA, axis=0)
        iPCA -= iMean
        iKernel = np.dot(iPCA, iPCA.T)
        #CF.append(np.sum(np.triu(iKernel, k=1))/structIdxs[i])
        CF.append(np.sum(np.triu(iKernel, k=0))/structIdxs[i])
        s += structIdxs[i]
        ss += 1
        if s >= len(batchPCA) and (f+1) < len(PCAFiles):
            f += 1
            if os.path.splitext(PCAFiles[f])[1] == '.npy':
                batchPCA = np.load(PCAFiles[f])[:, 0:nPCA]
            else:
                batchPCA = np.loadtxt(PCAFiles[f])[:, 0:nPCA]
            s = 0
        sys.stderr.write('Batch: %d, Structure: %d\r' % (f+1, ss))
    sys.stderr.write('\n')
    CF = np.asarray(CF)
    np.savetxt('CF.dat', CF)

def DEEM_score(DEEMFiles, IZAFiles, al, Z, nc=None, propName=None, m='euclidean'):
    if propName is not None:
        p = np.zeros(len(al))
    else:
        p = None
    nAtoms = np.zeros(len(al))
    volume = np.zeros(len(al))
    structIdxs = []
    for i, at in enumerate(al):
        n = 0
        atoms = at.get_atomic_numbers()
        for j in Z:
            n += np.count_nonzero(atoms == j)
        structIdxs.append(n)
        nAtoms[i] = len(atoms)
        volume[i] = np.linalg.det(at.cell)
        if propName is not None:
            p[i] = at.params[propName]

    volumes = np.repeat(volume, structIdxs)
    if propName is not None:
        ps = np.repeat(p, structIdxs)
    
    f = 0
    s = 0
    ss = 0
    score = []
    envscore = []
    mins = []
    envmin = []
    if os.path.splitext(DEEMFiles[f])[1] == '.npy':
        batchDEEM = np.load(DEEMFiles[f])[:, 0:nc]
    else:
        batchDEEM = np.loadtxt(DEEMFiles[f])[:, 0:nc]
    if os.path.splitext(IZAFiles[f])[1] == '.npy':
        batchIZA = np.load(IZAFiles[f])[:, 0:nc]
    else:
        batchIZA = np.loadtxt(IZAFiles[f])[:, 0:nc]

    for i in range(0, len(structIdxs)):
        iDEEM = batchDEEM[s:s+structIdxs[i]]
        dDEEM = cdist(iDEEM, batchIZA, metric=m)
        dMin = np.amin(dDEEM, axis=1)
        dMaxMin = np.amax(dMin)
        score.append(dMaxMin)
        envscore.append([dMaxMin]*structIdxs[i])
        mins.append(np.amin(dMin))
        envmin.append(dMin)
        s += structIdxs[i]
        ss += 1
        if s >= len(batchDEEM) and (f+1) < len(DEEMFiles):
            f += 1
            if os.path.splitext(DEEMFiles[f])[1] == '.npy':
                batchDEEM = np.load(DEEMFiles[f])[:, 0:nc]
            else:
                batchDEEM = np.loadtxt(DEEMFiles[f])[:, 0:nc]
            s = 0
        sys.stderr.write('Batch: %d, Structure: %d\r' % (f+1, ss))
    sys.stderr.write('\n')
    score = np.asarray(score)
    envscore = np.concatenate(envscore)
    mins = np.asarray(mins)
    envmin = np.concatenate(envmin)
    np.savetxt('maxmin.dat', score)
    np.savetxt('maxmins.dat', envscore)
    np.savetxt('mins.dat', mins)
    np.savetxt('envmin.dat', envmin)
    np.savetxt('volume.dat', volume)
    np.savetxt('volumes.dat', volumes)
    if propName is not None:
        np.savetxt('p.dat', p)
        np.savetxt('ps.dat', ps)

def vDist(v1, v2, kType):
    """
        Compute Euclidian or Manhattan norm
        between two vectors
        
        ---Arguments---
        v1, v2: vectors
        kType: norm type
    """
    if np.shape(v1) != np.shape(v2):
        sys.exit("Vectors are not same length")
    else:
        if kType == 'gaussian':
            return np.linalg.norm((v1-v2), ord=2)
        else:
            return np.linalg.norm((v1-v2), ord=1)

def matDist(mat1, mat2, kType):
    """
        Compute Euclidian or Manhattan norm
        between two matrices
        
        ---Arguments---
        mat1, mat2: matrices
        kType: norm type
    """
    if np.shape(mat1) != np.shape(mat2):
        sys.exit("Feature Matrices are not the same shape")
    else:
        if kType == 'gaussian':
            return np.linalg.norm((mat1-mat2).flatten(), ord=2)
        else:
            return np.linalg.norm((mat1-mat2).flatten(), ord=1)

def laplacianKernel(dMat, sigma):
    """
        Laplacian kernel
        
        ---Arguments---
        dMat: matrix of distances
        sigma: kernel width
    """
    return np.exp(-dMat/sigma)

def gaussianKernel(dMat, sigma):
    """
        Gaussian kernel
        
        ---Arguments---
        dMat: matrix of distances
        sigma: kernel width
    """
    return np.exp(-dMat**2/(2*sigma**2))

def MAE(predicted, observed):
    """
        Mean absolute error

        ---Arguments---
        predicted: vector of predicted property values
        observed: vector of observed (true) property values
    """
    if np.size(predicted) != np.size(observed):
        sys.exit("Predicted and observed vectors not same length")
    else:
        absErr = np.abs(predicted-observed)
        sup = np.amax(absErr)
        mae = np.sum(absErr)/np.size(predicted)
    return mae#, sup

def RMSE(predicted, observed):
    """
        Root mean square error

        ---Arguments---
        predicted: vector of predicted property values
        observed: vector of observed (true) property values
    """
    if np.size(predicted) != np.size(observed):
        sys.exit("Predicted and observed vectors not same length")
    else:
        rmse = np.sqrt(np.sum(np.power(predicted-observed, 2))/np.size(predicted))
    return rmse    

def get_random_structures(filename, nTotal, nRand):
    """
        Creates new xyz file comprising a random
        selection of structures from an input xyz file

        ---Arguments---
        filename: input file
        nTotal: total number of structures (in input file)
        nRand: number of structures to select randomly
    """
    sys.stderr.write('Selecting random structures...\n')
    randIdxs = random.sample(range(0, nTotal), nRand)
    structCount = -1
    headerCount = 0 
    randCount = 0
    g = open('randomSelection.xyz', 'w')
    
    with open(filename) as f:
        for line in f:
            
            # If the first part of the line is a number,
            # this indicates a new structure
            if line[0].isdigit() == True:
                structCount += 1
                headerCount += 1
                if structCount in randIdxs:
                    randCount += 1
                    doStruct = True
                    g.write(line)
                else:
                    doStruct = False
            else:
                if doStruct == True:
                    g.write(line)
                else:
                    continue
    g.close()

def do_FPS(x, D=0):
    """
        Farthest point sampling

        ---Arguments---
        x: input data to sample using FPS
        D: number of points to select
    """
    sys.stderr.write('Selecting FPS Points...\n')
    if D == 0:
        D = len(x)
    n = len(x)
    iy = np.zeros(D, np.int)

    # Faster evaluation of Euclidian Distance
    n2 = np.einsum("ai,ai->a", x, x)

    # Select first point at random
    iy[0] = np.random.randint(0, n)

    # Compute distances to all points
    dl = n2 + n2[iy[0]] - 2*np.dot(x, x[iy[0]])

    # Min max distances
    lmin = np.zeros(D)
    for i in range(1, D):
        iy[i] = np.argmax(dl)
        lmin[i-1] = dl[iy[i]]
        nd = n2 + n2[iy[i]] - 2*np.dot(x, x[iy[i]])
        dl = np.minimum(dl, nd)
        sys.stderr.write('Point: %d\r' % (i+1))
    sys.stderr.write('\n')
    np.savetxt('FPS.idxs', iy, fmt='%d')
    return iy

def quick_FPS(x, D=0, cutoff=1.0E-3):
    """
        "Quick" Farthest Point Sampling

        ---Arguments---
        x: input data to sample using quick FPS
        D: number of points to select
        cutoff: minimum standard deviation for selection
    """
    sys.stderr.write('Computing Quick FPS...\n')

    # Select components where standard deviation is greater than the cutoff
    # (selects most diverse components
    quickFPS = np.where(np.std(x, axis=0)/np.mean(np.std(x, axis=0)) > cutoff)[0]
    if D != 0:
        quickFPS = quickFPS[0:D]
    np.savetxt('quickFPS.idxs', quickFPS, fmt='%d')
    sys.stderr.write('Selected %d points\n' % len(quickFPS))
    return quickFPS

def randomSelect(x, D=0):
    """
        Select random points

        ---Arguments---
        x: input data to sample randomly
        D: number of points to select
    """
    idxs = range(0, len(x))
    np.random.shuffle(idxs)
    idxs = idxs[0:D]
    np.savetxt('random.idxs', idxs, fmt='%d')
    return idxs

def compute_SOAPs(al, d, idxs=None, batchSize=0, prefix='SOAPs'):
    """
        Compute SOAP vectors

        ---Arguments---
        al: Quippy AtomsList or AtomsReader
        d: Quippy descriptor
        idxs: list of indices to keep from the SOAP vector
        batchSize: number of structures to include in a batch
        prefix: prefix of output file
    """
    cwd = os.getcwd()
    sys.stderr.write('Computing SOAP vectors...\n')
    g = open('SOAPFiles.dat', 'w')
    SOAPFiles = []
    SOAPs = []
    if batchSize < 1:
        batchSize = len(al)
    N = 0
    for i, at in enumerate(al):
        at.set_cutoff(d.cutoff())
        at.calc_connect()
        SOAP = d.calc(at)['descriptor']
        if idxs is not None:
            SOAP = SOAP[:, idxs]
        SOAPs.append(SOAP)
        if ((i+1) % batchSize) == 0:
            SOAPs = np.concatenate(SOAPs)
            np.save(str('%s-%d' % (prefix, N)), SOAPs)
            g.write('%s/%s-%d.npy\n' % (cwd, prefix, N))
            SOAPs = []
            N += 1
        sys.stderr.write('Frame: %d\r' % (i+1))
    sys.stderr.write('\n')
    if len(SOAPs) > 0:
        SOAPs = np.concatenate(SOAPs)
        np.save(str('%s-%d' % (prefix, N)), SOAPs)
        g.write('%s/%s-%d.npy\n' % (cwd, prefix, N))
    g.close()

def build_covariance(SOAPFiles):
    """
        Iteratively builds covariance

        ---Arguments---
        SOAPFiles: list of files containing SOAP vectors in ASCII format
    """
    sys.stderr.write('Building covariance...\n')
    n = 0
    for i in SOAPFiles:
        with open(i, 'r') as f:
            for line in f:
                SOAP = np.asarray([float(x) for x in line.strip().split()])
                if n == 0:
                    p = np.shape(SOAP)[0]
                    SOAPMean = np.zeros(p)
                    C = np.zeros((p, p))
                n += 1
                C += (n-1)/float(n) * np.outer(SOAP-SOAPMean, SOAP-SOAPMean)
                SOAPMean = ((n-1)*SOAPMean + SOAP)/n 
                sys.stderr.write('Center: %d\r' % n)
    sys.stderr.write('\n')
    C = np.divide(C, n-1)
    sys.stderr.write('Saving covariance...\n')
    np.savetxt('cov.dat', C)
    sys.stderr.write('Saving mean...\n')
    np.savetxt('mean.dat', SOAPMean)

def build_PCA(C, nPCA):
    """
        Builds PCA from an input covariance matrix
        
        ---Arguments---
        C: covariance matrix
        nPCA: number of PCA components
    """
    sys.stderr.write('Building PCA...\n')
    p = np.shape(C)[0]
    u, V = np.linalg.eigh(C)
    u = np.flip(u, axis=0)
    V = np.flip(V, axis=1)
    D = np.zeros((p, p))
    g = np.zeros(p)
    for i in range(0, p):
        D[i, i] = u[i]
        g[i] = np.sum(D[0:i+1, 0:i+1])

    varRatio = g[0:nPCA]/g[-1]
    print "Variance Ratio", varRatio
    W = V[:, 0:nPCA]
    np.savetxt('eigenvectors.dat', W)
    np.savetxt('ratio.dat', varRatio)
    return W

def build_iPCA(SOAPFiles, nPCA, batchSize):
    """
        Builds PCA incrementally using SciKit Learn
        incremental PCA

        ---Arguments---
        SOAPFiles: list of files containing SOAP vectors
        nPCA: number of PCA components
        batchSize: batchSize for building the incremental PCA
    """
    sys.stderr.write('Building PCA...\n')
    PCABuilder = IncrementalPCA(n_components=nPCA, batch_size=batchSize)
    batch = []
    b = 0
    n = 0
    for idx, i in enumerate(SOAPFiles):
        SOAP = read_SOAP(i)
        PCABuilder.partial_fit(SOAP)
        sys.stderr.write('Batch: %d\r' % (idx+1))
    sys.stderr.write('\n')
    sys.stderr.write('Computing covariance...\n')
    C  = PCABuilder.get_covariance()
    sys.stderr.write('Saving covariance...\n')
    np.savetxt('cov.dat', C)
    sys.stderr.write('Saving mean...\n')
    np.savetxt('mean.dat', PCABuilder.mean_)
    sys.stderr.write('Saving eigenvectors...\n')
    np.savetxt('eigenvectors.dat', np.transpose(PCABuilder.components_))
    return PCABuilder

def transform_PCA(W, SOAPMean, SOAPFiles):
    """
        Transforms data according to the PCA

        ---Arguments---
        W: eigenvectors of the covariance matrix
        SOAPMean: mean of input data
        SOAPFiles: list of files containing SOAP vectors
    """
    cwd = os.getcwd()
    sys.stderr.write('Transforming PCA...\n')
    g = open('PCAFiles.dat', 'w')
    for idx, i in enumerate(SOAPFiles):
        #ct = 1
        SOAP = read_SOAP(i)
        transformedSOAP = np.inner(SOAP-SOAPMean, W.T)
        np.save(str('pca-%d' % idx), transformedSOAP)
        g.write('%s/pca-%d.npy\n' % (cwd, idx))
        sys.stderr.write('Batch: %d\r' % (idx+1)) 
    sys.stderr.write('\n')
    g.close()

def reconstruct_PCA(W, SOAPMean, PCAFiles, useRawData=False):
    """
        Reconstruct original data from PCA

        ---Arguments---
        W: eigenvectors of the covariance matrix
        SOAPMean: mean of the original data
        PCAFiles: list of files containing original data or PCA data
        useRawData: if True, reconstruct using original data; if False,
                    reconstruct using PCA data
        save: if True, save reconstructed data to file
    """
    cwd = os.getcwd()
    sys.stderr.write('Reconstructing data from PCA...\n')
    g = open('rSOAPFiles.dat', 'w')
    for idx, i in enumerate(PCAFiles):
        PCA = read_SOAP(i)
        if useRawData is True:
            transformedPCA = np.inner(PCA, np.inner(W, W)) + SOAPMean
        else:
            transformedPCA = np.inner(PCA, W) + SOAPMean
        np.save(str('rSOAP-%d' % idx), transformedPCA)
        g.write('%s/rSOAP-%d.npy\n' % (cwd, idx))
        sys.stderr.write('Batch: %d\r' % (idx+1)) 
    sys.stderr.write('\n')
    g.close()

def build_repSOAPs(inputFiles, repIdxs):
    repSOAPs = []
    n = 0
    sys.stderr.write('Building representative environments...\n')
    for idx, i in enumerate(inputFiles):
        iSOAP = read_SOAP(i)
        subIdxs = np.intersect1d(repIdxs[np.where(repIdxs >= n)],
                repIdxs[np.where(repIdxs < (n+len(iSOAP)))]) - n
        n += len(iSOAP)
        repSOAPs.append(iSOAP[subIdxs])
    repSOAPs = np.concatenate(repSOAPs)
    return repSOAPs

def read_input(SOAPFile):
    """
        Pre-processes list of files containing SOAP/SOAP-PCA data

        ---Arguments---
        SOAPFile: file containing file names and file paths to the
                  files containing the SOAP data

    """
    f = open(SOAPFile, 'r')
    inputFiles = f.readlines()
    inputFiles = [i.strip() for i in inputFiles]
    f.close()
    return inputFiles

def read_SOAP(SOAPFile):
    if os.path.splitext(SOAPFile)[1] == '.npy':
        iSOAP = np.load(SOAPFile)
    else:
        iSOAP = np.loadtxt(SOAPFile)
    return iSOAP

def center_data(SOAPFile):
    inputFiles = read_input(SOAPFile)
    n = 0
    dataMean = None
    for idx, i in enumerate(inputFiles):
        iSOAP = read_SOAP(i)
        if dataMean is None:
            dataMean = np.sum(iSOAP, axis=0)
        else:
            dataMean += np.sum(iSOAP, axis=0)
        n += len(iSOAP)

    dataMean /= n

    centeredFile = '%s-centered.dat' % os.path.splitext(SOAPFile)[0]
    f = open(centeredFile, 'w')
    for idx, i in enumerate(inputFiles):
        iSOAP = read_SOAP(i)
        iSOAP -=  dataMean
        np.save('%s-centered' % os.path.splitext(i)[0], iSOAP)
        f.write('%s-centered.npy\n' % os.path.splitext(i)[0])
    f.close()

def sparse_kPCA(inputFiles, repIdxs, kernel='gaussian', zeta=1, width=1.0, 
        nPCA=None, loadings=False, useRaw=False, lowmem=True):
    """
       Build and transform the kernel PCA

       ---Arguments---
       inputFiles: list of files containing the data
       repIdxs: indices of environments to use as the representatives
    """

    # Read inputfiles and build repSOAPs
    repSOAPs = build_repSOAPs(inputFiles, repIdxs)
    cwd = os.getcwd()
    f = open('KPCAFiles.dat', 'w')

    # Build kNM
    kNM = build_kernel_batch(inputFiles, repSOAPs,
            kernel=kernel, zeta=zeta, width=width, nc=None, lowmem=lowmem)

    # Build kMM
    kMM = build_kernel(repSOAPs, repSOAPs,
            kernel=kernel, zeta=zeta, width=width, nc=None)

    # Eigendecomposition on kMM
    w, U = np.linalg.eigh(kMM)
    w = np.flip(w, axis=0)
    U = np.flip(U, axis=1)

    # Take only positive eigenvalues
    w = w[np.where(w > 0)]
    U = U[:, 0:w.size]

    W = np.diagflat(1.0/np.sqrt(w))

    if lowmem is True:
        # Compute G
        P = np.dot(U, np.dot(W, U.T))
        Gmean = np.zeros(kMM.shape[0])
        n = 0
        for idx, i in enumerate(kNM):
            sys.stderr.write('Building approx kernel, batch %d\r' % (idx+1))
            kNMi = np.load('%s.npy' % i)
            Gi = np.dot(kNMi, P)
            Gmean += np.sum(Gi, axis=0)
            n += kNMi.shape[0]
            np.save('G%d' % idx, Gi)
        sys.stderr.write('\n')

        Gmean /= n

        G = np.zeros(kMM.shape)
        n = 0
        m = 0
        for idx, i in enumerate(kNM):
            sys.stderr.write('Centering approx. kernel, batch %d\r' % (idx+1))
            Gi = np.load('G%d.npy' % idx)
            Gi -= Gmean
            G += np.dot(Gi.T, Gi)
        sys.stderr.write('\n')

        # Eigendecomposition on (G.T)*G
        w, V = np.linalg.eigh(G)
        w = np.flip(w, axis=0)
        V = np.flip(V, axis=1)
        W = np.diagflat(1.0/w)

        # Approximate eigenvectors of kNN
        VW = np.dot(V, W)
        for idx, i in enumerate(kNM):
            sys.stderr.write('Building approx. eigenvectors '\
                    'and projecting, batch %d\r' % (idx+1))
            Gi = np.load('G%d.npy' % idx)
            Ui = np.dot(Gi, VW)
            
            # Retain desired number of principal components
            # and save the projections
            Ui = Ui[:, 0:nPCA]
            w = w[0:nPCA]

            if loadings is True:
                Gi = np.dot(Ui, np.diagflat(np.sqrt(w)))
            else:
                Gi = np.dot(Ui, np.diagflat(w))
            np.save('kpca-%d' % idx, Gi)
            f.write('%s/kpca-%d.npy\n' % (cwd, idx))
            os.system('rm G%d.npy k%d.npy' % (idx, idx))
        sys.stderr.write('\n')

    else:
        pass
        # Compute G
        sys.stderr.write('Building approx. kernel...\n')
        G = np.dot(kNM, np.dot(U, np.dot(W, U.T)))

        # Center G
        sys.stderr.write('Centering approx. kernel...\n')
        G -= np.mean(G, axis=0)

        # Eigendecomposition on (G.T)*G
        w, V = np.linalg.eigh(np.dot(G.T, G))
        w = np.flip(w, axis=0)
        V = np.flip(V, axis=1)
        W = np.diagflat(1.0/w)

        # Approximate eigenvectors of kNN
        sys.stderr.write('Building approx. eigenvectors...\n')
        U = np.dot(G, np.dot(V, W))

        # Retain desired number of principal components
        U = U[:, 0:nPCA]
        w = w[0:nPCA]

        # Projection
        sys.stderr.write('Projecting...\n')
        if loadings is True:
            np.save('kpca-0', np.dot(U, np.diagflat(np.sqrt(w))))
        else:
            np.save('kpca-0', np.dot(U, np.diagflat(w)))
        f.write('%s/kpca-0.npy\n' % cwd)

    f.close()

def npy_convert(fileList):
    """
        Converts from list of .npy files to ASCII

        ---Arguments---
        fileList: list of filenames to convert

    """
    for idx, i in enumerate(fileList):
        sys.stderr.write('Converting file: %d\r' % (idx+1))
        filename = os.path.splitext(i)[0]
        np.savetxt(str('%s.dat' % filename), np.load(i))
    sys.stderr.write('\n')

def npy_stdout(fileList):
    """
        Reads .npy file and prints to stdout

        ---Arguments---
        fileList: list of filenames to convert
    """
    n = 0
    for i in fileList:
        data = np.load(i)
        n += len(data)
        for j in data:
            print '    '.join(map(str, j))

def extract_structure_properties(al, Z, propName=None):
    """
        Extracts structue properties from xyz file

        ---Arguments---
        al: Quippy AtomsList or AtomsReader
        Z: central atom species
        propName: name of property to extract
    """
    p = np.zeros(len(al))
    nAtoms = np.zeros(len(al))
    volume = np.zeros(len(al))
    structIdxs = []
    for i, at in enumerate(al):
        n = 0
        atoms = at.get_atomic_numbers()
        for j in Z:
            n += np.count_nonzero(atoms == j)
        structIdxs.append(n)
        nAtoms[i] = len(atoms)
        volume[i] = np.linalg.det(at.cell)
        if propName != 'volume':
            p[i] = at.params[propName]
        else:
            p[i] = volume[i]

    return structIdxs, nAtoms, volume, p

def build_kernel_batch(inputFiles, SOAPs2, kernel='linear', 
        zeta=1, width=1.0, nc=None, lowmem=False):
    """
        SOAP kernel between two SOAP vectors in batches

        ---Arguments---
        inputFiles: list of filenames containing
                    SOAP vectors
        SOAPs2: input SOAP vectors
        zeta: exponent for nonlinear kernel
    """
    sys.stderr.write('Building kernel...\n')
    kList = []
    for idx, i in enumerate(inputFiles):
        SOAPs1 = read_SOAP(i)[:, 0:nc]
        if kernel == 'gaussian':
            d = cdist(SOAPs1, SOAPs2, metric='euclidean')
            k = gaussianKernel(d, width)
        elif kernel == 'laplacian':
            d = cdist(SOAPs1, SOAPs2, metric='cityblock')
            k = laplacianKernel(d, width)
        else:
            k = np.dot(SOAPs1, SOAPs2.T)**zeta
        if lowmem is True:
            np.save('k%d' % idx, k)
            kList.append('k%d' % idx)
        else:
            kList.append(k)
    if lowmem is True:
        k = kList
    else:
        k = np.concatenate(kList)
    return k

def build_kernel(SOAPs1, SOAPs2, kernel='linear', zeta=1, width=1.0, nc=None):
    """
        SOAP kernel between two SOAP vectors

        ---Arguments---
        SOAPs1, SOAPs2: input SOAP vectors
        zeta: exponent for nonlinear kernel
    """
    sys.stderr.write('Building kernel...\n')
    if kernel == 'gaussian':
        d = cdist(SOAPs1, SOAPs2, metric='euclidean')
        k = gaussianKernel(d, width)
    elif kernel == 'laplacian':
        d = cdist(SOAPs1, SOAPs2, metric='cityblock')
        k = laplacianKernel(d, width)
    else:
        k = np.dot(SOAPs1, SOAPs2.T)**zeta
    return k

def build_sum_kernel_batch(inputFiles, SOAPs2, structIdxs, kernel='linear', 
        zeta=1, width=1.0, nc=None):
    """
        Build sum kernel for a structure

        ---Arguments---
        inputFiles: list of filenames of files containing SOAP vectors 
        SOAPs2: input SOAP vectors
        structIdxs: list of indices indicating which
                    SOAP vectors belong to which structure
                    (output by extract_structure_properties)
        zeta: exponent for nonlinear kernel
        width: width for Gaussian or Laplacian kernel
        kernel: type of kernel to build
        nc: number of components to use
    """
    sys.stderr.write('Building sum kernel...\n')
    k = np.zeros((len(structIdxs), len(SOAPs2)))
    n = 0
    for i in inputFiles:
        SOAPs1 = read_SOAP(i)[:, 0:nc]
        m = 0
        while m < len(SOAPs1):
            iSOAPs1 = SOAPs1[m:m+structIdxs[n]]
            if kernel == 'gaussian':
                dj = cdist(iSOAPs1, SOAPs2, metric='euclidean')
                kj = gaussianKernel(dj, width)
                k[n, :] = np.sum(kj, axis=0)
            elif kernel == 'laplacian':
                dj = cdist(iSOAPs1, SOAPs2, metric='cityblock')
                kj = laplacianKernel(dj, width)
                k[n, :] = np.sum(kj, axis=0)
            else:
                k[n, :] = np.sum(np.dot(iSOAPs1, SOAPs2.T)**zeta, axis=0)
            m += structIdxs[n]
            n += 1
    return k

def build_sum_kernel(SOAPs1, SOAPs2, structIdxs, kernel='linear', 
        zeta=1, width=1.0):
    """
        Build sum kernel for a structure

        ---Arguments---
        SOAPs1, SOAPs2: input SOAP vectors
        structIdxs: list of indices indicating which
                    SOAP vectors belong to which structure
                    (output by extract_structure_properties)
        zeta: exponent for nonlinear kernel
    """
    sys.stderr.write('Building sum kernel...\n')
    k = np.zeros((len(structIdxs), len(SOAPs2)))
    n = 0
    for i in range(0, len(structIdxs)):
        iSOAP = SOAPs1[n:structIdxs[i]+n]
        if kernel == 'gaussian':
            dj = cdist(iSOAP, SOAPs2, metric='euclidean')
            kj = gaussianKernel(dj, width)
            k[i, :] = np.sum(kj, axis=0)
        elif kernel == 'laplacian':
            dj = cdist(iSOAP, SOAPs2, metric='euclidean')
            kj = gaussianKernel(dj, width)
            k[i, :] = np.sum(kj, axis=0)
        else:
            k[i, :] = np.sum(np.dot(iSOAP, SOAPs2.T)**zeta, axis=0)
        n += structIdxs[i]
    return k

def build_structure_kernel(SOAPs1, SOAPs2, structIdxs, zeta=1):
    """
        Build structural kernel

        ---Arguments---
        SOAPs1, SOAPs2: input SOAP vectors
        structIdxs: list of indices indicating which
                    SOAP vectors belong to which structure
                    (output by extract_structure_properties)
        zeta: exponent for nonlinear kernel
    """
    k = np.zeros((len(structIdxs), len(structIdxs)))
    n = 0
    for i in range(0, len(structIdxs)):
        m = 0
        iSOAP = SOAPs1[n:strucIdxs[i]+n]
        kRow = np.sum(np.dot(iSOAP, SOAPs2.T)**zeta, axis=0)
        for j in range(0, len(structIdxs)):
            k[i, j] = np.sum(kRow[m:structIdxs[j]+m], axis=1)
            m += structIdxs[j]
        n += structIdxs[i]
    return k

def property_regression(y, kMM, kNM, nStruct, idxsTrain, idxsValidate, 
                        sigma=1.0, jitter=1.0E-16, envKernel=None):
    """
        Perform property decomposition

        ---Arguments---
        y: structural property data
        kMM: kernel between representative environments
        kNM: sum kernel of representative environments
        SOAPs1, SOAPs2: full input SOAP vectors
        nStruct: number of structures
        idxsTrain: training indices
        idxsValidate: validation (or testing) indices
        sigma: regularization parameter
        jitter: value for additional regularization
        save: save regression data
        envKernel: compute environment decomposition
    """

    # Solve KRR problem
    delta = np.var(y)*len(kMM)/np.trace(kMM)
    K = kMM*delta*sigma**2 + np.dot(kNM[idxsTrain].T, kNM[idxsTrain])*delta**2
    maxEigVal = np.amax(np.linalg.eigvalsh(K))
    K += np.eye(len(kMM))*maxEigVal*jitter
    Y = delta*np.dot(delta*kNM[idxsTrain].T, y[idxsTrain])
    w = np.linalg.solve(K, Y)
    
    # Predict structure properties
    yy = np.dot(kNM, w)
    
    # Build environment kernel
    # need to use the build_kernel function here
    if envKernel is not None:
        if isinstance(envKernel, list):
            for idx, i in enumerate(envKernel):
                iKernel = read_SOAP('%s.npy'% i)
                yyEnv = np.dot(iKernel, w)
                np.savetxt('envProperties-%d.dat' % idx, yyEnv)
        else:

            # Decompose structural property into 
            # environmental contributions; save
            yyEnv = np.dot(envKernel, w)
            np.savetxt('envProperties.dat', yyEnv)

    return y[idxsTrain], y[idxsValidate], yy[idxsTrain], yy[idxsValidate]

def kernel_distance(ii, jj, kij):
    """
        Compute kernel induced distance

        ---Arguments---
        ii: Diagonal of K(A, A)
        jj: Diagonal of K(B, B)
        kij: K(A, B)
    """
    
    # ii and jj are diagonals of kii and kjj kernels
    # kij is kernel between i and j
    radicand = -2.0*kij + np.reshape(ii, (len(ii), 1)) + jj

    # Handle machine precision errors around 0
    radicand[np.where(radicand < 0.0)] = 0.0
    D = np.sqrt(radicand)
    return D

def kernel_histogram_rectangular(D, bins=200):
    """
        Compute histogram of kernel-induced distances
        for rectangular kernel

        ---Arguments---
        D: matrix of distances
        bins: number of histogram bins
    """
    H, binEdges = np.histogram(D.flatten(), bins=bins, density=True)
    return H, binEdges

def kernel_histogram_square(D, bins=200):
    """
        Compute histogram of kernel-induced distances
        for square kernel

        ---Arguments---
        D: matrix of distances
        bins: number of histogram bins
    """
    D = D[np.triu_indices(len(D))]
    H, binEdges = np.histogram(D, bins=bins, density=True)
    return H, binEdges

def kernel_histogram_min(D, bins=200, axis=None):
    """
        Compute histogram of minimum kernel-induced distances

        ---Arguments---
        D: matrix of distances
        bins: number of histogram bins
        axis: axis over which to minimize
    """
    D = np.amin(D, axis=axis)
    H, binEdges = np.histogram(D, bins=bins, density=True)
    return H, binEdges

