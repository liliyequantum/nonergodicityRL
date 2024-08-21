# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 12:49:07 2023

@author: admin
"""

import numpy as np
import math
import pickle
# from scipy.sparse import coo_matrix
# from scipy.sparse.linalg import svds
from tqdm import tqdm

import os
from scipy.io import savemat
import h5py #save -v7.3 .mat file
import hdf5storage

def update_dir(dirname):
    if not os.path.exists(dirname):
      os.mkdir(dirname)
      print("Folder %s created!" % dirname)
    else:
      print("Folder %s already exists" % dirname)
      
def generateBond(numLattice):
    bond = np.array([[0,numLattice-1],[0,1]]);
    for i in range(1,numLattice):
       
        if i==numLattice-1:
            bond = np.append(bond,[[i,i-1]],axis=0)
            bond = np.append(bond,[[i,0]],axis=0)
        else:
            bond = np.append(bond,[[i,i-1]],axis=0)
            bond = np.append(bond,[[i,i+1]],axis=0)
            
    return bond

def generateNeighborCoupling(bond,numLattice):
    neighborCoup = np.zeros((numLattice,numLattice));
    for i in range(0,len(bond)):
        neighborCoup[bond[i,0],bond[i,1]] = 1
    return neighborCoup

def count_bits_on_even_positions(number):
    count = 0
    position = 1
    
    while number > 0:
        if position % 2 == 1 and number & 1 ==1:
            count += 1
        number >>=1
        position += 1
    return count

def spinChainBasisIndex2Pattern(numLattice,numSpin):
    # complexity: 2**(#lattice)
    Index2Pattern = []
    num_even_bit = []
    for i in range(0,2**numLattice,1):
        char = '{0:0'+str(numLattice)+'b}' # fixed binary length, N
        binary_char = char.format(i)
        number_of_one = binary_char.count('1')
        if number_of_one == numSpin:
            Index2Pattern.append(binary_char)   
            num_even_bit.append(count_bits_on_even_positions(i))
    if numLattice<=10:
        print('chain_basis_pattern:',Index2Pattern)
    print('Spin_chain_basis_length:', len(Index2Pattern))
    return [Index2Pattern, num_even_bit]

def subChainBasisIndex2Pattern(numSubChain):
    subPattern = []
    for i in range(0,2**numSubChain,1):
        char = '{0:0'+str(numSubChain)+'b}' # fixed binary length, N
        binary_char = char.format(i)
        subPattern.append(binary_char)         
    
    Index2Pattern = []
    for i in range(len(subPattern)):
        for j in range(len(subPattern)):
            Index2Pattern.append(subPattern[i]+subPattern[j])
    
    if numSubChain<=5:
        print('sub_chain_basis_pattern:',Index2Pattern)
    print('sub_chain_basis_length:', len(Index2Pattern))
    return Index2Pattern

def basisPattern2Index(Index2Pattern):
    # complexity: #basis
    Pattern2Index = {}
    for i in range(0,len(Index2Pattern)):
        Pattern2Index[Index2Pattern[i]] = i
    return Pattern2Index

def mapSpinChain2subChain(upChainIndex_row,downChainIndex_col,numAChain,numBChain,upChainIndex2Pattern,\
    downChainIndex2Pattern,subAChainPattern2Index,subBChainPattern2Index):    
                             # spin up     #spin down

    # chainIndex_1,chainIndex_2,numHalfChain,chainIndex2Pattern,halfChainPattern2Index
    subAChainPattern_row = upChainIndex2Pattern[upChainIndex_row][:numAChain]+downChainIndex2Pattern[downChainIndex_col][:numAChain]
    subBChainPattern_col = upChainIndex2Pattern[upChainIndex_row][numAChain:]+downChainIndex2Pattern[downChainIndex_col][numAChain:]
    
    subAChainIndex_row = int(subAChainPattern2Index[subAChainPattern_row])
    subBChainIndex_col = int(subBChainPattern2Index[subBChainPattern_col])
    
    return [subAChainIndex_row, subBChainIndex_col]

def spinChain2subChain_row_col_table(numUpChainBasis,numDownChainBasis,numAChain,numBChain,upChainIndex2Pattern,\
    downChainIndex2Pattern,subAChainPattern2Index,subBChainPattern2Index):
 
    upChain_row = np.zeros(numUpChainBasis*numDownChainBasis,dtype=int)
    downChain_col =  np.zeros(numUpChainBasis*numDownChainBasis,dtype=int)
    subAChain_row = np.zeros(numUpChainBasis*numDownChainBasis,dtype=int)
    subBChain_col =  np.zeros(numUpChainBasis*numDownChainBasis,dtype=int)
    t = 0
    for i in tqdm(range(numUpChainBasis)): # row of mpsi, spin up
        for j in range(numDownChainBasis): # col of mpsi, spin down
            upChain_row[t] = i
            downChain_col[t] = j
            [subAChain_row[t], subBChain_col[t]] = mapSpinChain2subChain(i,j,numAChain,numBChain,upChainIndex2Pattern,\
                downChainIndex2Pattern,subAChainPattern2Index,subBChainPattern2Index)
            t = t+1
            
    return [upChain_row, downChain_col,subAChain_row ,subBChain_col]

def generateHhop(d,index2Pattern,pattern2Index,numLattice,neighborCoup,J):
    Hhop = np.zeros((d,d)) # HUphop(dUp,dUp),HDownhop(dDown,dDown)
    for basisIndex in tqdm(range(0,d)):
        # basisIndex=0
        basisPattern = index2Pattern[basisIndex]
        colIndex = basisIndex
        
        # print('basisPattern',basisPattern)
        for locLattice in range(0,numLattice):
            if basisPattern[locLattice]=='1':
                
                for locCoup in range(0,numLattice):
                    unoccupide = basisPattern[locCoup]=='0'
                    coupling = neighborCoup[locLattice,locCoup]==1.0
                    if  coupling & unoccupide:
                            
                            # print('locLattice',locLattice)
                            # print('locCoup',locCoup)
                            newbasisPattern = basisPattern
                            newbasisPattern = newbasisPattern[:locLattice]+'0'+newbasisPattern[(locLattice+1):]
                            newbasisPattern = newbasisPattern[:locCoup]+'1'+newbasisPattern[(locCoup+1):]
                            rowIndex = pattern2Index[newbasisPattern]
                            
                            # print('newbasisPattern',newbasisPattern)
                            if locLattice==0 & locCoup==numLattice-1:# sign maintains for the boundary hoppling
                                sign = 1
                            elif locLattice==numLattice-1 & locCoup==0:
                                sign = 1
                            else:
                                sign = numElectronPassSign(locLattice,locCoup,basisPattern)
                            Hhop[rowIndex,colIndex] = -J * sign
    return Hhop
    
    
def numElectronPassSign(locLattice,locCoup,basisPattern):
     if locLattice<locCoup:
         countElectron = basisPattern[locLattice:(locCoup+1)]
         numElectron = countElectron.count('1') - 1
     elif locLattice>locCoup:
         countElectron = basisPattern[locCoup:(locLattice+1)]
         numElectron = countElectron.count('1') - 1
     if numElectron %2 ==0:
         sign = 1
     else:
         sign = -1
        
     # print('numElectron',numElectron)
     # print('sign',sign)
     return sign

def Vpre(dUp,dDown,\
        upChainIndex2Pattern,downChainIndex2Pattern,numLattice):
   
    VpreUpOrDown = np.zeros((dUp,dDown))
    VpreUpAndDown = np.zeros((dUp,dDown))
    for basisUpIndex in tqdm(range(0,dUp)):
        for basisDownIndex in range(0,dDown):
            psiUp = upChainIndex2Pattern[basisUpIndex]
            psiDown = downChainIndex2Pattern[basisDownIndex]
            for loc in range(0, numLattice):
                upTrue = psiUp[loc]=='1'
                downTrue = psiDown[loc]=='1'
                if upTrue:
                    VpreUpOrDown[basisUpIndex,basisDownIndex] += loc
                    
                if downTrue:
                    VpreUpOrDown[basisUpIndex,basisDownIndex] += loc
  
                if upTrue & downTrue:
                    VpreUpAndDown[basisUpIndex,basisDownIndex] += 1
                                  
                       
    return VpreUpOrDown,VpreUpAndDown

# parameter
numLattice = 8 #numLattice<=14
numHalfChain = int(numLattice/2)

numUp = int(numLattice/2)
numDown = numUp

J = 1

print('basis for the up and down of whole chain')
[upChainIndex2Pattern, numUp_even_bit] = spinChainBasisIndex2Pattern(numLattice,numUp)
[downChainIndex2Pattern, numDown_even_bit] = spinChainBasisIndex2Pattern(numLattice,numDown)

upChainPattern2Index = basisPattern2Index(upChainIndex2Pattern)
downChainPattern2Index = basisPattern2Index(downChainIndex2Pattern)

if len(upChainIndex2Pattern)==math.factorial(numLattice)/math.factorial(numUp)/math.factorial(numLattice-numUp):
    numUpChainBasis = len(upChainIndex2Pattern)
if len(downChainIndex2Pattern)==math.factorial(numLattice)/math.factorial(numDown)/math.factorial(numLattice-numDown):
    numDownChainBasis = len(downChainIndex2Pattern)

print('basis for the half chain')

halfChainIndex2Pattern = subChainBasisIndex2Pattern(numHalfChain)
halfChainPattern2Index = basisPattern2Index(halfChainIndex2Pattern)

if len(halfChainIndex2Pattern)==2**(2*numHalfChain):
    numHalfChainBasis = len(halfChainIndex2Pattern)
    

print('index map from chain basis to half-chain basis of psi')
[upChain_row, downChain_col,halfChain_row,halfChain_col] = \
spinChain2subChain_row_col_table(numUpChainBasis,numDownChainBasis,\
                                      numHalfChain,numHalfChain,upChainIndex2Pattern,\
    downChainIndex2Pattern,halfChainPattern2Index,halfChainPattern2Index) 
    
# print('upChain',upChainIndex2Pattern[upChain_row[10]])
# print('downChain',downChainIndex2Pattern[downChain_col[10]])
# print('halfChain row',halfChainIndex2Pattern[halfChain_row[10]])
# print('halfChain col',halfChainIndex2Pattern[halfChain_col[10]])

print('bond')
bond = generateBond(numLattice)
neighborCoup = generateNeighborCoupling(bond,numLattice)
print('generate Hop')
HUphop = generateHhop(numUpChainBasis,upChainIndex2Pattern,upChainPattern2Index,\
                    numLattice,neighborCoup,J)
HDownhop = generateHhop(numDownChainBasis,downChainIndex2Pattern,downChainPattern2Index,\
                    numLattice,neighborCoup,J)
print('Vpre')
[VpreUpOrDown,VpreUpAndDown] = Vpre(numUpChainBasis,numDownChainBasis,\
                     upChainIndex2Pattern,downChainIndex2Pattern,numLattice)
     
rundir = './numLattice_'+str(numLattice)+'_numUp_'+str(numUp)+'_numDown_'+str(numDown)
update_dir(rundir)

savemat(rundir+'/data.mat',{'numLattice':numLattice,\
'numHalfChain':numHalfChain,\
'numUp':numUp,'numDown':numDown,\
'numUpChainBasis':numUpChainBasis,'numDownChainBasis':numDownChainBasis,\
'numHalfChainBasis':numHalfChainBasis,\
'numUp_even_bit':numUp_even_bit,'numDown_even_bit':numDown_even_bit,'halfChainIndex2Pattern':halfChainIndex2Pattern,\
'upChainIndex2Pattern' : upChainIndex2Pattern,'downChainIndex2Pattern' : downChainIndex2Pattern,\
'HUphop':HUphop,'HDownhop':HDownhop,'VpreUpOrDown':VpreUpOrDown,'VpreUpAndDown':VpreUpAndDown,\
'upChain_row':upChain_row, 'downChain_col':downChain_col, \
'halfChain_row':halfChain_row, 'halfChain_col':halfChain_col})