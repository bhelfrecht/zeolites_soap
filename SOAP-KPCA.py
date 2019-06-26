#!/usr/bin/python

import os
import sys
import argparse
import random
import quippy as qp
import numpy as np
import SOAPTools

parser = argparse.ArgumentParser()
parser.add_argument('-soap', type=str, default='SOAPFiles.dat', 
        help='File containing SOAP file filenames')
parser.add_argument('-pca', type=int, default=None, 
        help='Number of PCA components')
parser.add_argument('-kernel', type=str, default='linear',
        choices=['linear', 'gaussian', 'laplacian'], help='Kernel type')
parser.add_argument('-zeta', type=float, default=1, help='SOAP kernel zeta')
parser.add_argument('-width', type=float, default=1.0,
        help='Kernel width')
parser.add_argument('-lowmem', action='store_true',
        help='Low memory version of KPCA')
parser.add_argument('-idxs', type=str, default='FPS.idxs', 
        help='File with FPS indices for representative environments')
parser.add_argument('-dotransform', type=str, default=None,
        help='Project data based on existing kernel')
parser.add_argument('-output', type=str, default='.',
        help='Directory where the output files should be saved')

args = parser.parse_args()

### BUILD PCA ###
repIdxs = np.loadtxt(args.idxs, dtype=np.int)

inputFiles = SOAPTools.read_input(args.soap)

if args.dotransform is not None:
    testFiles = SOAPTools.read_input(args.dotransform)
    SOAPTools.sparse_kPCA_transform(inputFiles, testFiles, repIdxs,
            np.load('U.npy'),
            #np.load('P.npy'), np.load('V.npy'), np.load('Gmean.npy'),
            kernel=args.kernel, zeta=args.zeta, width=args.width,
            nPCA=args.pca, lowmem=args.lowmem, output=args.output)
else:
    SOAPTools.sparse_kPCA(inputFiles, repIdxs, kernel=args.kernel,
            zeta=args.zeta, width=args.width, nPCA=args.pca,
            lowmem=args.lowmem, output=args.output)
