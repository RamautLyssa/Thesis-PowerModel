import xlwings as xw

import matplotlib.pyplot as plt
import matplotlib
import math
import pandas as pd
import random
import numpy as np
import time
import os
import tempfile
import scipy.stats
import scipy.linalg as sl
from numpy import linalg as LA
import numpy.matlib
import pathlib
from scipy.linalg import toeplitz



sh1 = xw.Book('myproject.xlsm').sheets[0]
sh2 = xw.Book('myproject.xlsm').sheets[1]
sh3 = xw.Book('myproject.xlsm').sheets[2]


def distance(l, k,users_positions, station_positions): 
    return math.sqrt((users_positions[(k-1)][0] - station_positions[(l-1)][0])**2 + (users_positions[(k-1)][1] - station_positions[(l-1)][1])**2)
    
def calculation_Fkl(noiseFigure):  #in dB
    z_lk = np.random.normal(0, noiseFigure)
    return z_lk

def calculation_PLmk(l,k,users_positions, station_positions, L,sigma_sf): #in dB
    d = distance(l,k,users_positions, station_positions)
    #L = -140.7151
    if d > 50:
        PLlk = (-1*L) - 35*math.log10(d/1000) + (sigma_sf*np.random.randn())
        #print("1")
    elif 10 < d <= 50:
        PLlk = (-1*L) - 15*math.log10(50/1000) - 20*math.log10(d/1000)
        #print("2")
    elif d <= 10:
        PLlk = (-1*L) - 15*math.log10(50/1000) - 20*math.log10(10/1000) 
        #print("3")
    
    #DO we need to use L
    return PLlk


def horizontalBars(y, mylabels):
    fig = plt.figure()
    sizes_perc = [100*s/sum(y) for s in y]
    fig, ax = plt.subplots(figsize=(20,6)) 
    #22,12   
    y_h = np.arange(len(mylabels))
    ax.barh(mylabels, sizes_perc)
    ax.set_yticks(y_h)
    ax.set_yticklabels(mylabels)
    ax.set_ylabel('Percentage')
    ax.set_title("Verhouding power")
    
    for i, v in enumerate(sizes_perc):
        ax.text(v + 1, i, str(round(v,2))+'%', color='black', fontweight='bold') 

  
    ax.set_xlim(0, max(sizes_perc)+5)

	
    #with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
        #fig.savefig(tmpfile, format='png', dpi=600)
        #image_path = tmpfile.name

    # Insert image into worksheet using pictures.add
    sh1.pictures.add(image_path, name='MyPlot', update=True, left=sh1.range('D13').left + 200, top=sh1.range('D13').top)
    # Remove temporary file
    #os.unlink(image_path)

def SpectralEfficiencyUplink(expW, tauw_ul, tauw_c, eta_k, rho_ul, alfa,tauw_tr, rho_tr, sigma2_n,beta, Matrix):
    L = expW.shape[0]
    K = expW.shape[1]

    
    SE_ZF_UL = np.zeros(K)
    
    #Scale the square roots of power coefficients to satisfy all the per-AP power constraints
    #These coefficients correspond to the vectors\vect{\mu}_k in (8) in the paper
    #gammaEqual = gammaEqual*np.sqrt(Pmax)/np.max(np.linalg.norm(gammaEqual,axis=0))
    
    #Compute the SEs as in (6) in the paper
    for k in range(0,K):
        SINRnumerator = rho_ul*np.power(alfa,2)
        SINRdenominatorPartTEE = 0
        SINRdenominatorPartTN = 0
        SINRdenominatorPartQN = 0

        #AA - needs to be changed to M_k instead of L
        for ii in range(0,L):
            SINRdenominatorPartQN1 = 0
            for i in range(0,K):
                sumBeta = 0
                for iii in range(0,K):
                    sumBeta = sumBeta + beta[iii,ii] 
            
                SINRdenominatorPartTEE = SINRdenominatorPartTEE+ ((Matrix[ii][k]*expW[ii,k]*(beta[i,ii] - ((np.power(alfa,2)*tauw_tr*rho_tr* np.power(beta[i,ii],2))/((np.power(alfa,2)*(tauw_tr*rho_tr*beta[i,ii]+1)) + (sigma2_n*((rho_tr*sumBeta) + 1)))))))
                    
            SINRdenominatorPartTN = SINRdenominatorPartTN + (Matrix[ii][k]*expW[ii,k])
            SINRdenominatorPartQN = SINRdenominatorPartQN + (Matrix[ii][k]*expW[ii,k]*((rho_ul*sumBeta)+1)) 
                
                #np.matmul(gammaEqual[i:i+1,:].reshape(1,L), np.matmul(interference[:,:,k,i],gammaEqual[i:i+1,:].reshape(L,1)))
        SINR_TEE = np.power(alfa,2)*rho_ul*SINRdenominatorPartTEE
        SINR_TN = np.power(alfa,2)*SINRdenominatorPartTN
        SINR_QN = sigma2_n*SINRdenominatorPartQN
        #SE of one user k
        SE_ZF_UL[k] = (tauw_ul/tauw_c)*np.log2(1 + (SINRnumerator/(SINR_TEE + SINR_TN + SINR_QN)))
        #print("deler UL" + str((SINR_TEE + SINR_TN + SINR_QN)))
        #print("deler")
    return SE_ZF_UL

def SpectralEfficiencyDownlink(expV, tauw_dl, tauw_c, eta_k, rho_dl,alfa,tauw_tr,rho_tr,sigma2_n,beta, Matrix):
    L = expV.shape[0]
    K = expV.shape[1]
    
    SE_ZF_DL = np.zeros(K)
    
    #Scale the square roots of power coefficients to satisfy all the per-AP power constraints
    #These coefficients correspond to the vectors\vect{\mu}_k in (8) in the paper
    #gammaEqual = gammaEqual*np.sqrt(Pmax)/np.max(np.linalg.norm(gammaEqual,axis=0))
    
    #Compute the SEs as in (6) in the paper
    for k in range(0,K):
        SINRdenominatorPart = 0
        eta_k = 0 
        #for eta in the numerator
        for ll in range(0,L):
            etakaccent =0
            if (Matrix[ll][k] == 0):
                eta_k = eta_k + 0
            else:
                for kk in range(0,K):
                    etakaccent = etakaccent  + (Matrix[ll][kk]*(pow(beta[kk,ll],0.5)))
                eta_k = eta_k + ((pow(beta[k,ll],0.5))/etakaccent)
                #print(etakaccent)                
        
        #for eta in the denominator    
        for i in range(0,K):
            for ii in range(0,L):
                sumBeta = 0
                eta_lk = 0
                etakkaccent=0
                for iii in range(0,K):
                    sumBeta = sumBeta + beta[iii,ii]  
                
                if (Matrix[ii][i] == 0):
                    eta_lk =0
                else:
                    for kk in range(0,K):
                        etakaccent = etakaccent  + (Matrix[ii][kk]*(pow(beta[kk,ii],0.5)))
                    eta_lk = ((pow(beta[k,ii],0.5))/etakaccent)    
                    
                  
                SINRdenominatorPart = SINRdenominatorPart+ ((Matrix[ii][i]*expV[ii,i]*eta_lk*(beta[k,ii] - ((np.power(alfa,2)*tauw_tr*rho_tr* np.power(beta[k,ii],2))/((np.power(alfa,2)*(tauw_tr*rho_tr*beta[k,ii]+1)) + (sigma2_n*((rho_tr*sumBeta) + 1)))))))
            
        #np.matmul(gammaEqual[i:i+1,:].reshape(1,L), np.matmul(interference[:,:,k,i],gammaEqual[i:i+1,:].reshape(L,1)))
        #print("eta: "+ str(eta_k))
        SINRnumerator = rho_dl*eta_k
        #SE of one user k
        SE_ZF_DL[k] = (tauw_dl/tauw_c)*np.log2(1 + (SINRnumerator/(1+(rho_dl*SINRdenominatorPart))))
#         print("deler DL" + str((1+(rho_dl*SINRdenominatorPart))))
#         print("SINR denominatorPart: " + str(SINRdenominatorPart))
#         print("beta: " + str(beta[k,ii]))
#         print("teller van de deler: " + str((np.power(alfa,2)*tauw_tr*rho_tr* beta[k,ii])))
#         print("deler deel 1 van de deler: " +str((np.power(alfa,2)*(tauw_tr*rho_tr*beta[k,ii]+1))))
#         print("deler deel 2 van de deler:" + str((sigma2_n*((rho_tr*sumBeta) + 1))))
    return SE_ZF_DL

def SpectralEfficiencyDownlinkZonderAllocation(expV, tauw_dl, tauw_c, eta_k, rho_dl,alfa,tauw_tr,rho_tr,sigma2_n,beta, Matrix):
    L = expV.shape[0]
    K = expV.shape[1]
    
    SE_ZF_DL = np.zeros(K)
    eta_k = 1
    #Scale the square roots of power coefficients to satisfy all the per-AP power constraints
    #These coefficients correspond to the vectors\vect{\mu}_k in (8) in the paper
    #gammaEqual = gammaEqual*np.sqrt(Pmax)/np.max(np.linalg.norm(gammaEqual,axis=0))
    
    #Compute the SEs as in (6) in the paper
    for k in range(0,K):
        SINRdenominatorPart = 0
            
        for ii in range(0,L):
            for i in range(0,K):
                sumBeta = 0
                for iii in range(0,K):
                    sumBeta = sumBeta + beta[iii,ii]  
                
                 
                SINRdenominatorPart = SINRdenominatorPart+ ((Matrix[ii][i]*expV[ii,i]*eta_k*(beta[k,ii] - ((np.power(alfa,2)*tauw_tr*rho_tr* np.power(beta[k,ii],2))/((np.power(alfa,2)*(tauw_tr*rho_tr*beta[k,ii]+1)) + (sigma2_n*((rho_tr*sumBeta) + 1)))))))
                #np.matmul(gammaEqual[i:i+1,:].reshape(1,L), np.matmul(interference[:,:,k,i],gammaEqual[i:i+1,:].reshape(L,1)))
        SINRnumerator = rho_dl*eta_k
        #SE of one user k
        SE_ZF_DL[k] = (tauw_dl/tauw_c)*np.log2(1 + (SINRnumerator/(1+(rho_dl*SINRdenominatorPart))))
#         print("deler DL" + str((1+(rho_dl*SINRdenominatorPart))))
#         print("SINR denominatorPart: " + str(SINRdenominatorPart))
#         print("beta: " + str(beta[k,ii]))
#         print("teller van de deler: " + str((np.power(alfa,2)*tauw_tr*rho_tr* beta[k,ii])))
#         print("deler deel 1 van de deler: " +str((np.power(alfa,2)*(tauw_tr*rho_tr*beta[k,ii]+1))))
#         print("deler deel 2 van de deler:" + str((sigma2_n*((rho_tr*sumBeta) + 1))))
    return SE_ZF_DL


def functionRlocalscattering(M,theta,ASDdeg):
    ASD = ASDdeg*math.pi/180
    antennaSpacing = 0.5
    #The correlation matrix has a Toeplitz structure, so we only need to
    #compute the first row of the matrix
    firstRow = np.zeros((M,1), dtype='complex')
    
    for column in range(0,M):
    
        #Compute the approximated integral as in (2.24) in masterproef massive MIMO
        firstRow[column] = np.exp(1j*2*math.pi*antennaSpacing*math.sin(theta)*column)*np.exp(-ASD**2/2 * ( 2*math.pi*antennaSpacing*math.cos(theta)*column )**2)
    
    R = (toeplitz(firstRow)).transpose()
    return R


def sorted_SE(SE):
    
    K = SE.shape[0]
    nbrOfSetups = SE.shape[1]
    A=np.reshape(SE[:,0:nbrOfSetups],(K*nbrOfSetups,1))
    sorted_SE = A[A[:,0].argsort(kind='mergesort')]
    
    return (sorted_SE)





@xw.sub
def main():

   
    nbrOfSetups = 1
    nbrOfRealizations = 1

    sh1.range("A11").value = "Loading ..."
    ##fig = plt.figure()
    # Ask user for field dimensions
    field_length = int(sh1.range('B4').value) 
    field_width = int(sh1.range('B5').value) 
    ############################################################
    number_users = int(sh1.range('B6').value) #int(input("Enter the number of users: "))
    users_positions = []
    K = number_users
    ###################################
    #VALUES
    delta = sh1.range('E17').value
    T_coh = sh1.range('E4').value
    tauw_c = int(sh1.range('E5').value)
    tau_c = tauw_c
    zeta_dl= sh1.range('E6').value
    zeta_ul = sh1.range('E7').value
    tau_p =  K
    tauw_tr = tau_p
    B= sh1.range('E9').value
    tauw_ul = (tauw_c-tauw_tr)*zeta_ul
    tauw_dl = (tauw_c-tauw_tr)*zeta_dl
    T_slot = T_coh
    eta_AP = sh1.range('I16').value
    rho_tr = (sh1.range('E11').value)*1000
    rho_ul = (sh1.range('E12').value)*1000
    rho_dl = (sh1.range('E13').value)*1000
    P_cod = (sh1.range('E15').value)*0.000000001 #W/Gbit/s
    P_dec = (sh1.range('E16').value)*0.000000001 #W/Gbit/s
    N = int(sh1.range('B7').value)
    M = N
    P_fix = sh1.range('E10').value
    P_traffic = (sh1.range('E14').value)*0.000000001
 
    ###################################
    #Area porperties
    area = int(sh1.range('M11').value)
    for i in range(3):
        if(area==0):
            #Urban
            hAP = 20
            hU = 1.65
            f =2e3
            sigma_sf = 8
        if(area==1):
            #subUrban
            hAP = 20
            hU = 1.65
            f = 1.9e3
            sigma_sf = 8
        if(area==2):
            #Rural
            hAP = 40
            hU = 1.65
            f= 0.45e3
            sigma_sf = 8
    
    noiseFigure = 9
    L_f = (46.3 + 33.9*math.log10(f) - 13.82*math.log10(hAP) - (1.1*math.log10(f)-0.7)*hU + (1.56*math.log10(f)-0.8))
    #print("L_F" + str(L_f))
    noiseVariancedBm = -174 + 10*math.log10(B) + noiseFigure
    ####################################
    #Alfa_m
    bitm = int(sh1.range('I17').value)
    c = 1
    if bitm == 1:
        alpha = 0.6366
        c = 0
    elif bitm ==2:
        alpha = 0.8825
    elif bitm == 3: 
        alpha = 0.96546
    elif bitm == 4:
        alpha = 0.990503
    elif bitm == 5:
        alpha = 0.997501
    elif bitm > 5:
        alpha = 1 - (math.pi * math.sqrt(3)*pow(2,-2*bitm))/(2)
    sigma2_n = alpha*(1-alpha)
    
    eta_k = 1
    ###################################
    xMin=0;xMax=field_length
    yMin=0;yMax=field_width
    xDelta=xMax-xMin;yDelta=yMax-yMin #rectangle dimensions
    areaTotal=xDelta*yDelta
    lambdaAP = 40e-6
    station_positions = []

    #Simulate Poisson point process
    num_stations = scipy.stats.poisson(lambdaAP*areaTotal).rvs()#Poisson number of points
    if num_stations == 0:
        num_stations = 1
    APXpositions = xDelta*scipy.stats.uniform.rvs(0,1,((num_stations,1)))+xMin#x coordinates of Poisson points
    APYpositions = yDelta*scipy.stats.uniform.rvs(0,1,((num_stations,1)))+yMin#y coordinates of Poisson points
    station_positions = list(zip(APXpositions,APYpositions))
    L = num_stations
    
    #Plotting
    fig1 = plt.figure()
    plt.scatter(APXpositions,APYpositions, edgecolor='gold', facecolor='none', alpha=0.5,label="Access points"  )
    plt.xlabel("x [m]"); plt.ylabel("y [m]")
    
    #Printing
    print("The number of the access points is: ")
    print(num_stations)

    ############################################################
    #Angular standard deviation in the local scattering model (in degrees)
    ASDdeg = 10

    #Store identity matrix of size M x M
    eyeM = np.identity(M)
    #Equal power allocation
    SE_ZF_UL = np.zeros((K,nbrOfSetups))
    SE_ZF_DL = np.zeros((K,nbrOfSetups))
    SE_ZF_DLZ = np.zeros((K,nbrOfSetups))
    
    SE_ZF_DL_k = np.zeros(nbrOfSetups)
    SE_ZF_UL_k = np.zeros(nbrOfSetups)
    SE_ZF_DLZ_k = np.zeros(nbrOfSetups)
    SE_l_ul = np.zeros((L,nbrOfSetups))
    SE_ul = np.zeros(nbrOfSetups)
    SE_l_dl = np.zeros((L,nbrOfSetups))
    SE_dl = np.zeros(nbrOfSetups)
    SE_l_dlZ = np.zeros((L,nbrOfSetups))
    SE_dlZ = np.zeros(nbrOfSetups)
    L_A = np.zeros(nbrOfSetups)
    p = np.zeros(nbrOfSetups)
    SE_UL = 0
    SE_DL = 0
    SE_DLZ = 0
    #####################################
    #Prepare array for pilot indices of K UEs for all setups
    pilotIndex = np.zeros((K))
    # Datasets initializations for prediction
    dataset_expect_UL = np.zeros((L,K,nbrOfSetups))
    dataset_expect_DL = np.zeros((L,K,nbrOfSetups))

    for n in range(0,nbrOfSetups):
        print(n, 'setups out of', nbrOfSetups)
            
        UEpositions = np.zeros((K,1), dtype = 'complex')
        distances = np.zeros((K,L))

        #Prepare to store normalized spatial correlation matrices
        R = np.zeros((M,M,L,K), dtype = 'complex')

        #Prepare to store average channel gain numbers (in dB)
        channelGaindB = np.zeros((L,K))
        gainOverNoisedB = np.zeros((L,K))
        #Generate random UE locations together
        UEXpositions = []
        UEYpositions = []
        UEpositions = []

        for i in range(number_users):
            x = random.uniform(0, field_length)
            y = random.uniform(0, field_width)
            users_positions.append((x, y))
            UEXpositions.append((x,))
            UEYpositions.append((y,))
            UEpositions.append(complex(x, y))
            start = time.perf_counter()
            angletoUE = np.zeros((K,L))    

        UEXpositions = np.array(UEXpositions)
        UEYpositions = np.array(UEYpositions)
        UEpositions = np.array(UEpositions)

        #Plot field with stations
        x_positions, y_positions = zip(*users_positions)
        plt.plot(x_positions, y_positions, '*', color='r', label='active users')
        for i, position in enumerate(users_positions):
            plt.annotate(i+1, position)

        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        sh1.pictures.add(fig1, name='visualisation', update=True, left=sh1.range('A1').left, top=sh1.range('A1').top)
        
        for kk in range(0,K):
            Xdist = np.matlib.repmat(UEXpositions[kk,0], L, 1) - APXpositions
            Ydist = np.matlib.repmat(UEYpositions[kk,0], L, 1) - APYpositions
            for j in range(0,L):
                angletoUE[kk,j] = np.angle(Xdist[j] + 1j*Ydist[j])     
                R[:,:,j,kk] = functionRlocalscattering(M,angletoUE[kk,j],ASDdeg)
       

        Matrix_SUM_ALL_beta_lk = [0 for x in range(K)]
        Matrix_lk = [[0 for x in range(K)] for y in range(L)] 
        beta_lk = np.zeros((L,))
        
        betas = np.zeros((L,K))

        

        for k in range(K):
            for l in range(L):
                
                #betaSel[l,k] = pow(10,((calculation_PLmk(l,k,users_positions, station_positions, L_f)+ calculation_Fkl(noiseFigure))/10))
                gainOverNoisedB[l,k] = calculation_PLmk(l,k,users_positions, station_positions, L_f,sigma_sf) - noiseVariancedBm
             
                #add extra fading to all channels from APs to UE k that have a distance larger than 50 meters
                betas[l,k] = pow(10, (gainOverNoisedB[l,k]/10))
                
                Matrix_SUM_ALL_beta_lk[k]  =Matrix_SUM_ALL_beta_lk[k]+ betas[l,k]
                
            #print(" mean channelgainoverNoise" + str(np.mean(gainOverNoisedB[:,k])))
            #now we add this vector betas with size L to a numpy matrix as an extra row.
            beta_lk = np.vstack([beta_lk, betas[:,k]])
            #print("mean betas" + str(np.mean(betas[:,k])))
            # 2) make a tuple of it
            Matrix_beta_lk_tuple = [(i+1, value) for i, value in enumerate(betas[:,k])]
            # 3) sort this tuple on the values
            Sorted_matrix_beta_lk_tuple = sorted(Matrix_beta_lk_tuple, key=lambda x: x[1],reverse=True)
            #print(Sorted_matrix_beta_lk_tuple)
            # 4) calculate which APs serve this user K
            sum_AP = 0
            j = 0
            while sum_AP < delta:
                #Sorted_matrix_beta_lk_tuple[j]
                #So every selected AP has 'zero'
                result = Sorted_matrix_beta_lk_tuple[j][1]
                sum_AP = sum_AP + (result/Matrix_SUM_ALL_beta_lk[k])
                index = Sorted_matrix_beta_lk_tuple[j][0]
                Matrix_lk[index-1][k] = 1
                j = j+1
        beta_lk = beta_lk[1:]
        
        H = np.zeros((M*L,nbrOfRealizations,K), dtype = 'complex')
        CH = np.sqrt(0.5)*( np.random.randn(M*L,nbrOfRealizations,K)+1j*np.random.randn(M*L,nbrOfRealizations,K) )
        CorrR = np.zeros((M,M,L,K), dtype = 'complex')
        expect_UL = np.zeros((L,K), dtype = 'complex')
        expect_DL = np.zeros((L,K), dtype = 'complex')
        #Filling in H matrix
        for j2 in range(0,L):
            for k2 in range(0,K):      
                CorrR[:,:,j2,k2] =  betas[j2,k2]* R[:,:,j2,k2]
                Rsqrt = sl.sqrtm(CorrR[:,:,j2,k2])
                H[j2*N:(j2+1) * N, :, k2] = np.matmul(Rsqrt,(CH[j2*N :(j2+1) * N, :, k2]))
                #H[j2*N:(j2+1)*N, : , k2] = np.matmul(sqrt(betas[j2,k2]), (CH[j2*N :(j2+1) * N, :, k2]))
                #print("H" + str(H))

        pilotIndex = np.mod(np.random.permutation(K), tau_p)
    
        #Generate realizations of normalized noise
        Np = np.sqrt(0.5)*(np.random.randn(M,nbrOfRealizations,L,tau_p) + 1j*np.random.randn(M,nbrOfRealizations,L,tau_p))

        #Prepare to store results
        Hhat = np.zeros((L*M,nbrOfRealizations,K), dtype = 'complex')
        #Go through all APs
        for l in range(0, L):
            for t in range(0, tau_p):
                #Compute processed pilot signal for all UEs that use pilot t
                yp = np.sqrt(rho_tr*tau_p)* np.sum( H[(l*N):(l+1)*N,:,t==pilotIndex], 2 ) + Np[:,:,l,t]

                #Compute the matrix that is inverted in the MMSE estimator
                PsiInv = (rho_tr*tau_p* np.sum( CorrR[:,:,l,t==pilotIndex],2 ) + eyeM)

                #Go through all UEs that use pilot t
                for k in  np.argwhere(t==pilotIndex):
                    RPsi = np.matmul(CorrR[:,:,l,np.int(k)],np.linalg.inv(PsiInv))
                    Hhat[(l*N):(l+1)*N,:,np.int(k)] = np.sqrt(rho_tr*tau_p)*np.matmul( RPsi, yp )
        #print("Hhat" + str(Hhat))
            
        v_ZF = np.zeros((M,K,L), dtype = 'complex')
        w_ZF = np.zeros((M,K,L), dtype = 'complex')
        interf_RZF = np.zeros((K,K,L), dtype = 'complex')
        interf2_RZF = np.zeros((K,K,L), dtype = 'float')
        Hhatallj = np.zeros((M*L,K))
        
    
        for n1 in range(0,nbrOfRealizations):
            Hhatallj = Hhat[:, n1, :]
            W_ZF =  np.matmul(Hhatallj,np.linalg.pinv(np.matmul(np.conj(Hhatallj).T,Hhatallj)))
            #I don't use the normalized one. I use W_ZF. See later code.
            V_ZF = W_ZF/np.linalg.norm(W_ZF, axis = 0)
            
            for j4 in range(0,L):
                for k4 in range (0,K):
                    v = V_ZF[(j4*N):(j4+1)*N, k4]
                    #print(v.shape)
                    w = W_ZF[(j4*N):(j4+1)*N, k4]
                    #print(w.shape)
                    expect_UL[j4,k4] = expect_UL[j4,k4] + (np.power(np.linalg.norm(w),2))/nbrOfRealizations
                    expect_DL[j4,k4] = expect_DL[j4,k4] + (np.power(np.linalg.norm(v),2))/nbrOfRealizations

        expect_UL = np.abs(expect_UL)
        #print("expect_UL" + str(expect_UL))
        expect_DL = np.abs(expect_DL)
        dataset_expect_UL[:,:,n] = expect_UL
        dataset_expect_DL[:,:,n] = expect_DL
        SE_ZF_DL[:,n] = SpectralEfficiencyDownlink(expect_UL,tauw_dl,tauw_c, eta_k, rho_dl,alpha, tauw_tr,rho_tr,sigma2_n,betas.transpose(), Matrix_lk)
        SE_ZF_DLZ[:,n] = SpectralEfficiencyDownlinkZonderAllocation(expect_UL,tauw_dl,tauw_c, eta_k, rho_dl,alpha, tauw_tr,rho_tr,sigma2_n,betas.transpose(), Matrix_lk)
        SE_ZF_UL[:, n] = SpectralEfficiencyUplink(expect_UL, tauw_ul, tauw_c, eta_k, rho_ul, alpha,tauw_tr, rho_tr, sigma2_n,betas.transpose(), Matrix_lk)
        #print("SE DL voor user k is equal to " + str(SE_ZF_DL[:,n]))
        #print("SE UL voor user k is equal to " + str(SE_ZF_UL[:,n]))

        SE_ZF_DL_k[n] = np.sum(SE_ZF_DL[:,n])
        SE_ZF_DLZ_k[n] = np.sum(SE_ZF_DLZ[:,n])
        SE_ZF_UL_k[n] = np.sum(SE_ZF_UL[:,n])
#         print("SE DL voor SUM all k users is equal to " + str(SE_ZF_DL_k[n]))
#         print("SE DL (zonder power allocation) voor SUM all k users is equal to " + str(SE_ZF_DLZ_k[n]))
#         print("SE UL voor SUM all k users is equal to " + str(SE_ZF_UL_k[n]))
        


        #een matrix moet gemaakt worden die per AP zegt, Ja of Nee active
        for l in range(L):
            active = 0
            for k in range(K):
                SE_l_ul[l,n] = SE_l_ul[l,n] + (Matrix_lk[l][k]*SE_ZF_UL[k, n])
                SE_l_dl[l,n] = SE_l_dl[l,n] + (Matrix_lk[l][k]*SE_ZF_DL[k, n])
                SE_l_dlZ[l,n] = SE_l_dlZ[l,n] + (Matrix_lk[l][k]*SE_ZF_DLZ[k,n])
                if (Matrix_lk[l][k] == 1):
                    active = 1

            SE_ul[n] = SE_ul[n] + (active*SE_l_ul[l,n])
            SE_dl[n] = SE_dl[n] + (active*SE_l_dl[l,n])
            SE_dlZ[n] = SE_dlZ[n] + (active*SE_l_dlZ[l,n])
    
        df = pd.DataFrame(Matrix_lk)
        L_A[n] = (df.sum(axis=1) > 0).sum()
        
        
        pp = 0
        for l in range(0,L):
            aactive = 0
            p_l = 0
            for k in range(0,K):
                p_lk = 0
                etakaccent=0
                if (Matrix_lk[l][k] == 0):
                    p_lk =0
                else:
                    for kk in range(0,K):
                        etakaccent = etakaccent  + (Matrix_lk[l][kk]*(pow(betas[l,kk],0.5)))
                    p_lk = (rho_dl/1000)*((pow(betas[l,k],0.5))/etakaccent)    
                p_l = p_l + (Matrix_lk[l][k]*p_lk)
                if (Matrix_lk[l][k] == 1):
                    aactive = 1
            pp = pp + (aactive*p_l)
            
        p[n] = pp
        

    #is niet iets realisitsch voor ons model!
    #print("SE DL voor SUM all k users, average, is " + str(np.sum(SE_ZF_DL_k)/nbrOfSetups))
    #print("SE DL (zonder power allocation) voor SUM all k users, average, is " + str(np.sum(SE_ZF_DLZ_k)/nbrOfSetups)) 
    #print("SE UL voor SUM all k users, average, is " + str(np.sum(SE_ZF_UL_k)/nbrOfSetups))  
    
    SE_UL = np.mean(SE_ul)
    SE_DL =np.mean(SE_dl)
    SE_DLZ = np.mean(SE_dlZ)
    
    #CpuChoice = 1: Xeon Phi 'KNC'Intel 5110P
    #CpuChoice = 2: GTX Titan 'Kepler' NVIDIA GK110
    #CpuChoice = 3: Arndale CPU 'Cortex-A15' Samsung Exynos 5

    CpuChoice = 2
    if (CpuChoice==1):
        name = "Xeon Phi 'KNC'Intel 5110P"
        #Xeon Phi 'KNC'Intel 5110P (22nm)
        #P_idle = sh1.range('I5').value
        #L_{cu} = sh1.range('I7').value
        P_idle = int(sh1.range('I5').value)
        L_cu = int(sh1.range('I7').value)
        flops_4nodes = int(sh1.range('I6').value)
        E_l1 = int(sh1.range('I8').value)
        E_l2 = int(sh1.range('I9').value)
        E_MEM = int(sh1.range('I10').value)
        X_l1 = int(sh1.range('I11').value)
        X_l2 = int(sh1.range('I12').value)
        X_l3 = int(sh1.range('I13').value)
            
    if (CpuChoice==2): 
        name = "GTX Titan 'Kepler' NVIDIA GK110"
        #Graphics coprocessor "GTX Titan 'Kepler' NVIDIA GK110 (28nm)"
        P_idle = int(sh1.range('I5').value)
        L_cu = int(sh1.range('I7').value)
        flops_4nodes = int(sh1.range('I6').value)
        E_l1 = int(sh1.range('I8').value)
        E_l2 = int(sh1.range('I9').value)
        E_MEM = int(sh1.range('I10').value)
        #https://www.techpowerup.com/gpu-specs/geforce-gtx-titan.c1996
        X_l1 = int(sh1.range('I11').value)
        X_l2 = int(sh1.range('I12').value)
        X_l3 = int(sh1.range('I13').value)

    if (CpuChoice==3):
        name = "Arndale CPU 'Cortex-A15' Samsung Exynos 5"
        #Arndale CPU 'Cortex-A15' Samsung Exynos 5 (32nm)
        P_idle = int(sh1.range('I5').value)
        L_cu = int(sh1.range('I7').value)
        flops_4nodes = int(sh1.range('I6').value)
        E_l1 = int(sh1.range('I8').value)
        E_l2 = int(sh1.range('I9').value)
        E_MEM = int(sh1.range('I10').value)
        #https://cdrinfo.com/d7/content/samsung-offers-developers-arndale-board
        X_l1 = int(sh1.range('I11').value)
        X_l2 = int(sh1.range('I12').value)
        X_l3 = int(sh1.range('I13').value)


    
    

    
    sorted_SE_ZF_DL = sorted_SE(SE_ZF_DL)
    sorted_SE_ZF_UL = sorted_SE(SE_ZF_UL)
    fig = plt.figure(figsize=(16,12))
    plt.plot(sorted_SE_ZF_UL,np.linspace(0,1,K*nbrOfSetups),color='green',linewidth=4)
    plt.xlabel('Spectral efficiency UL [bit/s/Hz]')
    plt.ylabel('CDF')
    fig.savefig("SE_UL_10000km_10000km(20UE, 10N)_CPU_GTX.png", bbox_inches='tight', dpi=800)
    sh3.pictures.add(fig, name='Spectral efficiency UL', update=True, left=sh3.range('A1').left, top=sh3.range('A1').top)

    fig = plt.figure(figsize=(16,12))
    plt.plot(sorted_SE_ZF_DL,np.linspace(0,1,K*nbrOfSetups),color='blue',linewidth=4)
    plt.xlabel('Spectral efficiency DL [bit/s/Hz]')
    plt.ylabel('CDF')
    fig.savefig("SE_DL_10000km_10000km(20UE, 10N)_CPU_GTX.png", bbox_inches='tight', dpi=800)
    sh3.pictures.add(fig, name='Spectral efficiency DL', update=True, left=sh3.range('A1').left, top=sh3.range('L1').top)


    
    num_cols = len(df.columns)
    col_names = [f"UE {i+1}" for i in range(num_cols)]
    df.columns = col_names
    
    num_rows = len(df.index)
    row_names = [f"AP {i+1}" for i in range(num_rows)]
    df.index = row_names
    
    L_nonA = L
    L = int(np.mean(L_A))
    L_A = L
    
    sh2.clear_contents()
    table = 'Selected APs'
    #sh2["A1"].options(pd.DataFrame, header=1, index=True, expand='table').value = df
    sh1.range("A11").value = "Done! You can find more information on the other sheets."
    if table in [table.name for table in sh2.tables]:
        sh2.tables[table].update(df)
    else:
        mytable = sh2.tables.add(source=sh2['A1'], name=table,table_style_name='TableStyleMedium12').update(df)
    

    
	
    C_ce = 4*K*N*L*tauw_tr + 2*N*K*L*tauw_tr + 2*N*K*L*(tauw_tr-1) + 4*K*L*pow(N,2) + 2*K*L*pow(N,2) + 2*K*L*(N-1)
    C_precRec = 12*pow(K,2)*L*N + 8*K*L*N + (4/3)*pow(N,3) + 3*pow(N,2) -(10/3)*N - pow(K,2) - K
    C_LP = 8*L*N*K -2*L*N

    N_flops = (C_ce + C_LP + C_precRec)/(T_coh)
    N_4nodes = math.ceil(N_flops/flops_4nodes)

    w = 12
    ### Total memory needed
    N_mem_bits = 6*w*K*L*N + 2*w*K*K #bits
    N_mem_bytes = N_mem_bits/8
    #first we place everything that we can in L1 
    if (N_mem_bytes < X_l1):
        N_l1 = 1
        N_l2 = 0
        N_MEM = 0
    elif (X_l1 <N_mem_bytes<(X_l2+X_l1)):
        N_l1 = X_l1/N_mem_bytes
        N_l2 = (N_mem_bytes-X_l1)/N_mem_bytes
        N_MEM = 0
    elif ((X_l2+X_l1)<N_mem_bytes):
        N_l1 = X_l1/N_mem_bytes
        N_l2 = X_l2/N_mem_bytes
        N_MEM = (N_mem_bytes -X_l1 - X_l2)/N_mem_bytes

    ### Number of accesses
    data = tauw_ul + tauw_dl
    #print("Het aantal symbols that needs to be precoded/recombined: " + str(data))
    N_mem_ce = (N*tauw_tr) + (tauw_tr) + (N*tauw_tr*tauw_tr)
    N_mem_pre = 2*L*N*pow(K,2) + pow(K,2) + (2*L*K*N) + K + pow(K,2) 
    N_mem_lp = (L*N*K*data)
    N_mem_access = N_mem_ce + N_mem_pre + N_mem_lp
    ### Power for memory access 
    P_MEM = (N_l1*((N_mem_access*((2*w)/8)*E_l1)/(T_slot/2))) + (N_l2*((N_mem_access*((2*w)/8)*E_l2)/(T_slot/2))) + (N_MEM*((N_mem_access*((2*w)/8)*E_MEM)/(T_slot/2)))




    P_tx =(tauw_dl*np.mean(p))/(tauw_c*eta_AP)

    P_FH_LI= L_A* P_fix

    
    P_AP = (22.5e-3 + N*(36e-3+(c*4e-3)+(6e-5*pow(2,alpha))))*L_A
    P_ce = C_ce*(B/(tauw_c*L_cu))
    P_precRec= C_precRec*(B/(tauw_c*L_cu))
    P_LP = C_LP*(B/(L_cu)) * (1-((tauw_tr)/tauw_c))
    
    P_CU = P_idle * N_4nodes

    P_FH_LD_ul = B*P_traffic*SE_UL
    P_FH_LD_dl = B* P_traffic*SE_DL
    P_FH_LD = P_FH_LD_ul + P_FH_LD_dl
    
    P_coding= B*P_cod*SE_DL
    P_decoding = B*P_dec*SE_UL
    P_cod_dec = P_coding + P_decoding
    
    TotalPower = P_tx+ P_FH_LI + P_FH_LD + P_AP + P_ce + P_precRec + P_LP + P_MEM + P_CU + P_cod_dec

    print("Total power [W] = " + str(TotalPower))
    sh1.range("B13").value = TotalPower
    print("P_tx [W] = " + str(P_tx))
    sh1.range("B14").value = P_tx
    print("P_FH_LI [W] = " + str(P_FH_LI))
    sh1.range("B15").value = P_FH_LI
    print("P_FH_LD [W] = " + str(P_FH_LD))
    sh1.range("B16").value = P_FH_LD
    print("P_AP [W] = " + str(P_AP))
    sh1.range("B17").value = P_AP
    print("P_CE [W] = " + str(P_ce))
    sh1.range("B18").value = P_ce
    print("P_precRec [W] = " + str(P_precRec))
    sh1.range("B19").value = P_precRec
    print("P_LP [W] = " + str(P_LP))
    sh1.range("B20").value = P_LP
    print("P_MEM [W] = " + str(P_MEM))
    sh1.range("B21").value = P_MEM
    print("P_CU [W] =" + str(P_CU))
    sh1.range("B22").value = P_CU
    print("P_cod_dec [W] =" + str(P_cod_dec))
    sh1.range("B23").value = P_cod_dec

    sh1.range("B25").value = num_stations
    sh1.range("B26").value = L_A
    sh1.range("B28").value = N_flops
    sh1.range("B29").value = N_4nodes

    mylabels = ["Transmit power", "Load independent fronthaul power","Load dependent fronthaul power", "Power consumption APs", "Power consumption channel estimation", "Power consumption prec/recomb", "Power consumption LP", "Power consumption memory access", "Power consumption CU", "Power consumption encoding/decoding"]
    y = [P_tx, P_FH_LI,  P_FH_LD, P_AP, P_ce, P_precRec, P_LP, P_MEM, P_CU, P_cod_dec]   
    #x = range(5,15)
    #yy = [P_tx, P_FH_LI, P_FH_LD, P_AP, P_ce , P_precRec, P_LP, P_MEM, P_CU]
    #horizontalBars(y, mylabels)
    sizes_perc = [100*s/TotalPower for s in y]
    fig, ax = plt.subplots()        
    y_h = np.arange(len(mylabels))
    ax.barh(mylabels, sizes_perc, color='gold')
    ax.set_yticks(y_h)
    ax.set_yticklabels(mylabels)
    ax.set_ylabel('Different power terms')
    ax.set_title("Power consumption [%]")
    for i, v in enumerate(sizes_perc):
        ax.text(v + 1, i, str(round(v,2))+'%', color='black', fontweight='bold') 
    sh1.pictures.add(fig, name='MyPlot', update=True, left=sh1.range('D13').left, top=sh1.range('D13').top)
        
if __name__ =="__main__":
	xw.Book("myproject.xlsm").set_mock_caller()
	main()


