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
        rmse = np.sqrt(np.sum(np.power(predicted-observed, 2)) \
                /np.size(predicted))
    return rmse    

def get_random_structures(filename, nTotal, nRand, output='.'):
    """
        Creates new xyz file comprising a random
        selection of structures from an input xyz file

        ---Arguments---
        filename: input file
        nTotal: total number of structures (in input file)
        nRand: number of structures to select randomly
        output: output directory
    """

    # Prepare random indices and output file
    sys.stdout.write('Selecting random structures...\n')
    randIdxs = random.sample(range(0, nTotal), nRand)
    structCount = -1
    headerCount = 0 
    randCount = 0
    g = open('%s/randomSelection.xyz' % output, 'w')
    
    with open(filename) as f:
        for line in f:
            
            # If the first part of the line is a number,
            # this indicates a new structure
            if line[0].isdigit() == True:
                structCount += 1
                headerCount += 1

                # If this is a selected structure,
                # write the number of atoms
                # to the xyz file
                if structCount in randIdxs:
                    randCount += 1
                    doStruct = True
                    g.write(line)
                else:
                    doStruct = False

            else:

                # Write the comment line
                # and atom info
                if doStruct == True:
                    g.write(line)
                else:
                    continue
    g.close()

def do_FPS(x, D=0, output='.'):
    """
        Farthest point sampling

        ---Arguments---
        x: input data to sample using FPS
        D: number of points to select
        output: output directory
    """
    sys.stdout.write('Selecting FPS Points...\n')
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

        # Aggregate data point indices
        iy[i] = np.argmax(dl)

        # Aggregate distances
        lmin[i-1] = dl[iy[i]]

        # Compute new distances
        nd = n2 + n2[iy[i]] - 2*np.dot(x, x[iy[i]])
        dl = np.minimum(dl, nd)

        sys.stdout.write('Point: %d\r' % (i+1))
        sys.stdout.flush()

    sys.stdout.write('\n')

    # Write output
    np.savetxt('%s/FPS.idxs' % output, iy, fmt='%d')
    return iy

def quick_FPS(x, D=0, cutoff=1.0E-3, output='.'):
    """
        "Quick" Farthest Point Sampling

        ---Arguments---
        x: input data to sample using quick FPS
        D: number of points to select
        cutoff: minimum standard deviation for selection
        output: output directory
    """
    sys.stdout.write('Computing Quick FPS...\n')

    # Select components where standard deviation is greater than the cutoff
    # (selects most diverse components
    quickFPS = np.where(np.std(x, axis=0)/np.mean(np.std(x, axis=0)) > cutoff)[0]

    # Retain D components
    if D != 0:
        quickFPS = quickFPS[0:D]

    # Save output
    np.savetxt('%s/quickFPS.idxs' % output, quickFPS, fmt='%d')
    sys.stdout.write('Selected %d points\n' % len(quickFPS))
    return quickFPS

def randomSelect(x, D=0, output='.'):
    """
        Select random points

        ---Arguments---
        x: input data to sample randomly
        D: number of points to select
        output: output directory
    """

    # Select random indices
    idxs = range(0, len(x))
    np.random.shuffle(idxs)

    # Retain D indices
    idxs = idxs[0:D]

    # Save output
    np.savetxt('%s/random.idxs' % output, idxs, fmt='%d')
    return idxs

def compute_SOAPs(al, d, idxs=None, batchSize=0, 
        prefix='SOAPs', output='.'):
    """
        Compute SOAP vectors

        ---Arguments---
        al: Quippy AtomsList or AtomsReader
        d: Quippy descriptor
        idxs: list of indices to keep from the SOAP vector
        batchSize: number of structures to include in a batch
        prefix: prefix of output file
        output: output directory
    """
    sys.stdout.write('Computing SOAP vectors...\n')

    # Open file for output and initialize containers
    # for the SOAP vectors
    g = open('%s/SOAPFiles.dat' % output, 'w')
    SOAPFiles = []
    SOAPs = []

    # If the batchSize is invalid or not specified,
    # assume one batch
    if batchSize < 1:
        batchSize = len(al)

    # Batch count
    N = 0

    # Loop through the atoms list
    for i, at in enumerate(al):

        # Set the calculation cutoff
        # and calculate the SOAP vector
        at.set_cutoff(d.cutoff())
        at.calc_connect()
        SOAP = d.calc(at)['descriptor']

        # Retain the specified indices
        if idxs is not None:
            SOAP = SOAP[:, idxs]

        SOAPs.append(SOAP)

        # Once we reach the limit of the batch size,
        # write output
        if ((i+1) % batchSize) == 0:
            SOAPs = np.concatenate(SOAPs)
            np.save('%s/%s-%d' % (output, prefix, N), SOAPs)
            g.write('%s/%s-%d.npy\n' % (os.path.abspath(output), prefix, N))
            SOAPs = []
            N += 1

        sys.stdout.write('Frame: %d\r' % (i+1))
        sys.stdout.flush()

    sys.stdout.write('\n')

    # If we reach the end of the atoms list
    # and still have some structures that
    # we haven't written yet, write them
    if len(SOAPs) > 0:
        SOAPs = np.concatenate(SOAPs)
        np.save('%s/%s-%d' % (output, prefix, N), SOAPs)
        g.write('%s/%s-%d.npy\n' % (os.path.abspath(output), prefix, N))

    g.close()

def build_iPCA(SOAPFiles, nPCA, batchSize, output='.'):
    """
        Builds PCA incrementally using SciKit Learn
        incremental PCA

        ---Arguments---
        SOAPFiles: list of files containing SOAP vectors
        nPCA: number of PCA components
        batchSize: batchSize for building the incremental PCA
        output: output directory
    """
    sys.stdout.write('Building PCA...\n')

    # Initialize the PCA
    PCABuilder = IncrementalPCA(n_components=nPCA, batch_size=batchSize)

    # Initialize the batches
    batch = []
    b = 0
    n = 0

    # Go through all batches
    for idx, i in enumerate(SOAPFiles):

        # Read data
        SOAP = read_SOAP(i)

        # Build the PCA
        PCABuilder.partial_fit(SOAP)
        sys.stdout.write('Batch: %d\r' % (idx+1))
        sys.stdout.flush()

    sys.stdout.write('\n')

    # Compute covariance
    sys.stdout.write('Computing covariance...\n')
    C  = PCABuilder.get_covariance()
    sys.stdout.write('Saving covariance...\n')
    np.savetxt('%s/cov.dat' % output, C)

    # Compute mean
    sys.stdout.write('Saving mean...\n')
    np.savetxt('%s/mean.dat' % output, PCABuilder.mean_)

    # Save eigenvectors
    sys.stdout.write('Saving eigenvectors...\n')
    np.savetxt('%s/eigenvectors.dat' % output, 
            np.transpose(PCABuilder.components_))
    return PCABuilder

def transform_PCA(W, SOAPMean, SOAPFiles, output='.'):
    """
        Transforms data according to the PCA

        ---Arguments---
        W: eigenvectors of the covariance matrix
        SOAPMean: mean of input data
        SOAPFiles: list of files containing SOAP vectors
        output: output directory
    """
    sys.stdout.write('Transforming PCA...\n')
    g = open('%s/PCAFiles.dat' % output, 'w')

    # Go over all batches and project the data
    # using the provided mean and eigenvectors
    for idx, i in enumerate(SOAPFiles):
        #ct = 1
        SOAP = read_SOAP(i)
        transformedSOAP = np.inner(SOAP-SOAPMean, W.T)
        np.save('%s/pca-%d' % (output, idx), transformedSOAP)
        g.write('%s/pca-%d.npy\n' % (os.path.abspath(output), idx))
        sys.stdout.write('Batch: %d\r' % (idx+1)) 
        sys.stdout.flush()

    sys.stdout.write('\n')
    g.close()

def reconstruct_PCA(W, SOAPMean, PCAFiles, useRawData=False, output='.'):
    """
        Reconstruct original data from PCA

        ---Arguments---
        W: eigenvectors of the covariance matrix
        SOAPMean: mean of the original data
        PCAFiles: list of files containing original data or PCA data
        useRawData: if True, reconstruct using original data; if False,
                    reconstruct using PCA data
        output: output directory
    """
    sys.stdout.write('Reconstructing data from PCA...\n')
    g = open('%s/rSOAPFiles.dat' % output, 'w')

    # Go over all batches and reconstruct the data
    for idx, i in enumerate(PCAFiles):
        PCA = read_SOAP(i)

        # Reconstruct using original (raw) data (e.g., SOAP vectors)
        if useRawData is True:
            transformedPCA = np.inner(PCA, np.inner(W, W)) + SOAPMean

        # Reconstruct using already transformed, PCA data
        else:
            transformedPCA = np.inner(PCA, W) + SOAPMean

        # Write output
        np.save('%s/rSOAP-%d' % (output, idx), transformedPCA)
        g.write('%s/rSOAP-%d.npy\n' % (os.path.abspath(output), idx))
        sys.stdout.write('Batch: %d\r' % (idx+1)) 
        sys.stdout.flush()

    sys.stdout.write('\n')
    g.close()

def build_repSOAPs(inputFiles, repIdxs):
    """
        Build list of representative SOAP environments
        
        ---Arguments---
        inputFiles: list of files containing the SOAP data
        repIdxs: indices for the representative environments
    """

    # Initialize list of representatives and
    # a count of the total number of environments
    repSOAPs = []
    n = 0
    sys.stdout.write('Building representative environments...\n')

    # Loop over the batches
    for idx, i in enumerate(inputFiles):
        iSOAP = read_SOAP(i)

        # Since the data is in batches, the index of the
        # environments in iSOAP won't correspond to the 
        # environments that we actually need, so we keep
        # a count of the total number of environments (n).
        # Adding the index of an environment in iSOAP to 
        # n gives the corresponding index in repIdxs.
        subIdxs = np.intersect1d(repIdxs[np.where(repIdxs >= n)],
                repIdxs[np.where(repIdxs < (n+len(iSOAP)))]) - n
        n += len(iSOAP)
        repSOAPs.append(iSOAP[subIdxs])

    # Concatenate SOAPs into a single matrix
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
    """
        Read the SOAP data, automatically
        using load or loadtxt based
        on the file extension

        ---Arguments---
        SOAPFile: filename of the file with the SOAP data

    """
    if os.path.splitext(SOAPFile)[1] == '.npy':
        iSOAP = np.load(SOAPFile)
    else:
        iSOAP = np.loadtxt(SOAPFile)
    return iSOAP

def center_data(SOAPFile):
    """
        Read and center the batched data

        ---Arguments---
        SOAPFile: file containing file names and file paths
                  to the files containing the SOAP data
    """

    # Get list of filenames containing the SOAP data
    inputFiles = read_input(SOAPFile)

    # Initialize counter of the number of environments
    # and the mean
    n = 0
    dataMean = None

    # Loop through batches and iteratively
    # compute the mean
    for idx, i in enumerate(inputFiles):
        iSOAP = read_SOAP(i)

        # The first batch
        if dataMean is None:
            dataMean = np.sum(iSOAP, axis=0)

        # Subsequent batches
        else:
            dataMean += np.sum(iSOAP, axis=0)

        n += len(iSOAP)

    dataMean /= n

    # Prepare the output file
    centeredFile = '%s-centered.dat' % os.path.splitext(SOAPFile)[0]
    f = open(centeredFile, 'w')

    # Center each batch of data and write to file
    for idx, i in enumerate(inputFiles):
        iSOAP = read_SOAP(i)
        iSOAP -=  dataMean
        np.save('%s-centered' % os.path.splitext(i)[0], iSOAP)
        f.write('%s-centered.npy\n' % os.path.splitext(i)[0])

    f.close()

def sparse_kPCA_transform(inputFilesTrain, inputFilesTest, repIdxs, U,
        kernel='gaussian', zeta=1, width=1.0, nPCA=None, lowmem=True, output='.'):
    """
        Transform the kernel PCA
        FOR TESTING ONLY
        Arguments in brackets do not appear to be used/needed
        in this testing function

        ---Arguments---
        inputFilesTrain: files containing file names and paths
                         to the KPCA training data
        inputFilesTest: files containing file names and paths
                         to the out-of-sample that we wish to project
                         onto the KPCA constructed on the training data
        (repIdx)s: indices of representative environments in the training set
        U: KPCA eigenvectors
        kernel: kernel to use
        zeta: zeta for linear (dot product) kernel
        width: width for Gaussian/Laplacian kernel
        (nPCA): number of KPCA components to use
        (lowmem): perform computation in batches to preserve RAM
        (output): output directory
    """

    # Read the input data
    train_SOAP = read_SOAP(inputFilesTrain[0])
    test_SOAP = read_SOAP(inputFilesTest[0])
    n = train_SOAP.shape[0]
    l = test_SOAP.shape[0]

    # Kernel between OOS and the training data
    kLN = build_kernel(test_SOAP, train_SOAP,
            kernel=kernel, zeta=zeta, width=width, nc=None)

    # Kernel between all environments in the training set
    kNN = build_kernel(train_SOAP, train_SOAP,
            kernel=kernel, zeta=zeta, width=width, nc=None)

    # Center the kernel, based on: 
    # https://www.ics.uci.edu/~welling/classnotes/papers_class/Kernel-PCA.pdf
    L = np.ones((l, 1))/n
    N = np.ones((1, n))/n
    M1 = np.dot(np.sum(kLN, axis=1).reshape((l, 1)), N)
    M2 = np.dot(L, np.sum(kNN, axis=1).reshape((1, n)))
    M3 = np.dot(L, np.dot(np.sum(kNN), N))
    kLN -= M1 + M2 + M3

    # Save the KPCA projections
    np.savetxt('kpca_transform.dat', np.dot(kLN, U))

def sparse_kPCA(inputFiles, repIdxs, kernel='gaussian', zeta=1, width=1.0, 
        nPCA=None, lowmem=True, output='.'):
    """
       Build and transform the kernel PCA

       ---Arguments---
       inputFiles: list of files containing the data
       repIdxs: indices of environments to use as the representatives
       kernel: kernel to use in the KPCA construction
       zeta: zeta for linear kernel
       width: width for Gaussian/Laplacian kernel
       nPCA: Number of PCA components to use
       lowmem: Compute KPCA in batches to save RAM
       output: output directory

       The procedure is adapted from that stated in the SAS/IML 15.1
       user manual: https://documentation.sas.com/?docsetId=imlug&docsetTarget=imlug_langref_sect226.htm&docsetVersion=15.1&locale=en
    """

    # Read inputfiles and build repSOAPs
    repSOAPs = build_repSOAPs(inputFiles, repIdxs)
    f = open('%s/KPCAFiles.dat' % output, 'w')

    # Build kNM
    kNM = build_kernel_batch(inputFiles, repSOAPs,
            kernel=kernel, zeta=zeta, width=width, nc=None, 
            lowmem=lowmem, output=output)

    # Build kMM
    kMM = build_kernel(repSOAPs, repSOAPs,
            kernel=kernel, zeta=zeta, width=width, nc=None)

    # Eigendecomposition on kMM
    w, U = np.linalg.eigh(kMM)
    w = np.flip(w, axis=0)
    U = np.flip(U, axis=1)

    # Take only positive eigenvalues
    w = w[w > 0]
    U = U[:, 0:w.size]

    W = np.diagflat(1.0/np.sqrt(w))

    if lowmem is True:
        # Compute G
        P = np.dot(U, np.dot(W, U.T))
        # np.save('P', P)

        # Initialize the mean and data point count
        Gmean = np.zeros(kMM.shape[0])
        n = 0

        # Loop over all kernel batches
        for idx, i in enumerate(kNM):
            sys.stdout.write('Building approx kernel, batch %d\r' % (idx+1))
            sys.stdout.flush()

            # Load the kernel
            kNMi = np.load('%s.npy' % i)

            # Compute G, increment the sum of G,
            # and increment the data point count
            Gi = np.dot(kNMi, P)
            Gmean += np.sum(Gi, axis=0)
            n += kNMi.shape[0]

            # Save the G for this batch
            np.save('%s/G%d' % (output, idx), Gi)

        sys.stdout.write('\n')

        # Compute the mean of G
        Gmean /= n
        # np.save('Gmean', Gmean)

        G = np.zeros(kMM.shape)
        n = 0
        m = 0

        # Center G
        for idx, i in enumerate(kNM):
            sys.stdout.write('Centering approx. kernel, batch %d\r' % (idx+1))
            sys.stdout.flush()
            Gi = np.load('%s/G%d.npy' % (output, idx))
            Gi -= Gmean
            G += np.dot(Gi.T, Gi)

        sys.stdout.write('\n')

        # Eigendecomposition on (G.T)*G
        w, V = np.linalg.eigh(G)
        # np.save('V', V) 
        w = np.flip(w, axis=0)
        V = np.flip(V, axis=1)
        W = np.diagflat(1.0/w)
        uout = open('%s/U.npy' % output, 'w')

        # Approximate eigenvectors of kNN
        VW = np.dot(V, W)
        for idx, i in enumerate(kNM):
            sys.stdout.write('Building approx. eigenvectors '\
                    'and projecting, batch %d\r' % (idx+1))
            Gi = np.load('%s/G%d.npy' % (output, idx))
            Ui = np.dot(Gi-Gmean, VW)
            
            # Retain desired number of principal components
            # and save the projections
            Ui = Ui[:, 0:nPCA]
            w = w[0:nPCA]
            np.save(uout, Ui) # To get all of the batches, np.load
                              # must be called several times, 
                              # on a file object opened with 'rb', as one call
                              # will load one batch. Kind of a pain,
                              # but we don't really need this file anyway
                              # (this procedure does not appear to work in Python 3)

            # Store the KPCA projection in G and write
            Gi = np.dot(Ui, np.diagflat(w))
            np.save('%s/kpca-%d' % (output, idx), Gi)
            f.write('%s/kpca-%d.npy\n' % (os.path.abspath(output), idx))

            # Remove intermediate files
            os.system('rm %s/G%d.npy %s/k%d.npy' % (output, idx, output, idx))

        sys.stdout.write('\n')

    # Use this routine for computing the KPCA in a single batch
    # It has not been tested as extensively as the lowmem
    # version, as we always just use the lowmem version anyway
    else:
        pass
        # Compute G
        sys.stdout.write('Building approx. kernel...\n')
        G = np.dot(kNM, np.dot(U, np.dot(W, U.T)))

        # Center G
        sys.stdout.write('Centering approx. kernel...\n')
        G -= np.mean(G, axis=0)

        # Eigendecomposition on (G.T)*G
        w, V = np.linalg.eigh(np.dot(G.T, G))
        w = np.flip(w, axis=0)
        V = np.flip(V, axis=1)
        W = np.diagflat(1.0/w)

        # Approximate eigenvectors of kNN
        sys.stdout.write('Building approx. eigenvectors...\n')
        U = np.dot(G, np.dot(V, W))

        # Retain desired number of principal components
        U = U[:, 0:nPCA]
        w = w[0:nPCA]

        # Projection
        sys.stdout.write('Projecting...\n')
        np.save('%s/U' % output, U)
        np.save('%s/kpca-0' % output, np.dot(U, np.diagflat(w)))
        f.write('%s/kpca-0.npy\n' % os.path.abspath(output))

    f.close()

def npy_convert(fileList):
    """
        Converts from list of .npy files to ASCII
        and stores in a file

        ---Arguments---
        fileList: list of filenames to convert

    """
    for idx, i in enumerate(fileList):
        sys.stdout.write('Converting file: %d\r' % (idx+1))
        filename = os.path.splitext(i)[0]
        np.savetxt('%s.dat' % filename, np.load(i))
    sys.stdout.write('\n')

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

    # Property values for the structures
    p = np.zeros(len(al))

    # Numbers of atoms in the structures
    nAtoms = np.zeros(len(al))

    # Cell volumes of the structures
    volume = np.zeros(len(al))

    # Initialize storage for the
    # number of central atoms in each structure
    structIdxs = []

    # Loop over the structures in the atoms list
    for i, at in enumerate(al):
        n = 0

        # Get list of atomic numbers in the structure
        atoms = at.get_atomic_numbers()

        # For each central atom species,
        # get the number of central atoms
        for j in Z:
            n += np.count_nonzero(atoms == j)

        # Append central atom count
        structIdxs.append(n)

        # Append total number of atoms
        nAtoms[i] = len(atoms)

        # Append cell volume
        volume[i] = np.linalg.det(at.cell)

        # Append additional property
        if propName != 'volume':
            p[i] = at.params[propName]
        else:
            p[i] = volume[i]

    return structIdxs, nAtoms, volume, p

def build_kernel_batch(inputFiles, SOAPs2, kernel='linear', 
        zeta=1, width=1.0, nc=None, lowmem=False, output='.'):
    """
        SOAP kernel between two SOAP vectors in batches

        ---Arguments---
        inputFiles: list of filenames containing
                    SOAP vectors
        SOAPs2: input SOAP vectors (e.g., representatives
                or a test set)
        zeta: exponent for dot product (linear) kernel
        width: width for Gaussian/Laplacian kernel
        nc: number of components to use
        lowmem: compute (and store) the kernel in batches
        output: output directory
    """
    sys.stdout.write('Building kernel...\n')

    # Initialize list of batched kernels
    kList = []

    # Loop over each batch
    for idx, i in enumerate(inputFiles):

        # Read the SOAPs in the batch
        SOAPs1 = read_SOAP(i)[:, 0:nc]

        # Build the desired kernel
        if kernel == 'gaussian':
            d = cdist(SOAPs1, SOAPs2, metric='euclidean')
            k = gaussianKernel(d, width)
        elif kernel == 'laplacian':
            d = cdist(SOAPs1, SOAPs2, metric='cityblock')
            k = laplacianKernel(d, width)
        else:
            k = np.dot(SOAPs1, SOAPs2.T)**zeta

        # Save the kernel if we use lowmem
        # and append the filename to kList
        if lowmem is True:
            np.save('%s/k%d' % (output, idx), k)
            kList.append('%s/k%d' % (output, idx))

        # Append the kernel itself to kList
        # if we don't use lowmem
        else:
            kList.append(k)

    # If lowmem is True, return the list of filenames
    # for the files containing the batched kernel matrices.
    # Otherwise, return the (concatenated) kernel matrix
    if lowmem is True:
        k = kList
    else:
        k = np.concatenate(kList)
    return k

def build_kernel(SOAPs1, SOAPs2, kernel='linear', zeta=1, width=1.0, nc=None):
    """
        SOAP kernel between two SOAP vectors (single batch)

        ---Arguments---
        SOAPs1, SOAPs2: input SOAP vectors
        kernel: kernel to use
        zeta: exponent for nonlinear kernel
        width: width for Gaussian/Laplacian kernel
        nc: number of components to use (ignored)
    """
    sys.stdout.write('Building kernel...\n')

    # Compute Euclidean distances between data points
    # (if needed) and compute the desired kernel
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
        Build sum kernel for a structure in batches

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
    sys.stdout.write('Building sum kernel...\n')
    k = np.zeros((len(structIdxs), len(SOAPs2)))
    n = 0

    # Loop over the batches
    for i in inputFiles:
        SOAPs1 = read_SOAP(i)[:, 0:nc]
        m = 0

        # A batch can contain several structures,
        # so we use m and n to keep track of
        # which environments in the batch
        # belong to which structure.
        # The ordering of the data vectors
        # is the same as the atom ordering
        # in the xyz files
        while m < len(SOAPs1):

            # SOAP vectors for structure n
            iSOAPs1 = SOAPs1[m:m+structIdxs[n]]

            # Build the desired kernel
            if kernel == 'gaussian':
                dj = cdist(iSOAPs1, SOAPs2, metric='euclidean')
                kj = gaussianKernel(dj, width)

                # Store the sum kernel for structure n
                k[n, :] = np.sum(kj, axis=0)
            elif kernel == 'laplacian':
                dj = cdist(iSOAPs1, SOAPs2, metric='cityblock')
                kj = laplacianKernel(dj, width)

                # Store the sum kernel for structure n
                k[n, :] = np.sum(kj, axis=0)
            else:

                # Store the sum kernel for structure n
                k[n, :] = np.sum(np.dot(iSOAPs1, SOAPs2.T)**zeta, axis=0)

            # Increment the environment numbers
            # and the structure numbers
            m += structIdxs[n]
            n += 1

    return k

def build_sum_kernel(SOAPs1, SOAPs2, structIdxs, kernel='linear', 
        zeta=1, width=1.0):
    """
        Build sum kernel for a structure (single batch)

        ---Arguments---
        SOAPs1, SOAPs2: input SOAP vectors. SOAPs2 serves
                        as the "representative environments"
        structIdxs: list of indices indicating which
                    SOAP vectors belong to which structure
                    (output by extract_structure_properties)
        kernel: kernel type to build
        zeta: exponent for nonlinear kernel
        width: width for Gaussian or Laplacian kernel
    """
    sys.stdout.write('Building sum kernel...\n')
    k = np.zeros((len(structIdxs), len(SOAPs2)))
    n = 0

    # Loop over the structures
    for i in range(0, len(structIdxs)):

        # Load the data for structure i
        iSOAP = SOAPs1[n:structIdxs[i]+n]

        # Build the desired kernel
        if kernel == 'gaussian':
            dj = cdist(iSOAP, SOAPs2, metric='euclidean')
            kj = gaussianKernel(dj, width)

            # Store the sum kernel for structure i
            k[i, :] = np.sum(kj, axis=0)
        elif kernel == 'laplacian':
            dj = cdist(iSOAP, SOAPs2, metric='euclidean')
            kj = gaussianKernel(dj, width)

            # Store the sum kernel for structure i
            k[i, :] = np.sum(kj, axis=0)
        else:

            # Store the sum kernel for structure i
            k[i, :] = np.sum(np.dot(iSOAP, SOAPs2.T)**zeta, axis=0)

        # Increment environment numbers
        n += structIdxs[i]

    return k

def property_regression(y, kMM, kNM, nStruct, idxsTrain, idxsValidate, 
                        sigma=1.0, jitter=1.0E-16, 
                        envKernel=None, output='.'):
    """
        Perform property decomposition

        ---Arguments---
        y: structural property data
        kMM: kernel between representative environments
        kNM: sum kernel of representative environments
        nStruct: number of structures
        idxsTrain: training indices
        idxsValidate: validation (or testing) indices
        sigma: regularization parameter
        jitter: value for additional regularization
        envKernel: compute environment decomposition
        output: output directory
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
    
    # Compute environment decomposition if an environment
    # kernel is passed
    if envKernel is not None:

        # Environment kernel is stored in batches
        if isinstance(envKernel, list):
            for idx, i in enumerate(envKernel):
                iKernel = read_SOAP('%s.npy'% i)
                yyEnv = np.dot(iKernel, w)
                np.savetxt('%s/envProperties-%d.dat' % (output, idx), yyEnv)

        # Environment kernel is a single array
        else:

            # Decompose structural property into 
            # environmental contributions; save
            yyEnv = np.dot(envKernel, w)
            np.savetxt('%s/envProperties.dat' % output, yyEnv)

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

    # The distance matrix is symmetric, so we only
    # need to look at half of it
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

    # Compute minimum distance over the specified axis
    D = np.amin(D, axis=axis)
    H, binEdges = np.histogram(D, bins=bins, density=True)
    return H, binEdges

