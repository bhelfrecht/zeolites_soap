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
parser.add_argument('-loadings', action='store_true',
        help='Calculate KPCA with loadings')
parser.add_argument('-raw', action='store_true',
        help='Transform the PCA with real data kernel '\
                'instead of approximated kernel (not currently implemented')

args = parser.parse_args()

### BUILD PCA ###
repIdxs = np.loadtxt(args.idxs, dtype=np.int)

inputFiles = SOAPTools.read_input(args.soap)

SOAPTools.sparse_kPCA(inputFiles, repIdxs, kernel=args.kernel,
        zeta=args.zeta, width=args.width, nPCA=args.pca,
        loadings=args.loadings, useRaw=args.raw, lowmem=args.lowmem)

