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



sh1 = xw.Book(r'C:\Users\Windows 10\Desktop\myproject\myproject.xlsm').sheets[0]
sh2 = xw.Book(r'C:\Users\Windows 10\Desktop\myproject\myproject.xlsm').sheets[1]
sh3 = xw.Book(r'C:\Users\Windows 10\Desktop\myproject\myproject.xlsm').sheets[2]


def distance(l, k,users_positions, station_positions): 
    return math.sqrt((users_positions[(k-1)][0] - station_positions[(l-1)][0])**2 + (users_positions[(k-1)][1] - station_positions[(l-1)][1])**2)
    
def calculation_Fkl():  #in dB
    sigma = 4
    z_lk = np.random.normal(0, pow(sigma,2))
    return z_lk

def calculation_PLmk(l,k,users_positions, station_positions, L): #in dB
    d = distance(l,k,users_positions, station_positions)
    if d > 50:
        PLlk = L - 35*math.log(10, d)
        #print("1")
    elif 10 < d <= 50:
        PLlk = L - 15*math.log10(50) - 20*math.log10(d)
        #print("2")
    elif d <= 10:
        PLlk = L - 15*math.log(10, 50) - 20*math.log(10, 10) 
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

@xw.sub
def main():

   
    start = time.time()

    sh1.range("A11").value = "Loading ..."
    ##fig = plt.figure()
    # Ask user for field dimensions
    field_length = int(sh1.range('B4').value) 
    #float(input("Enter the length of the field: "))
    field_width = int(sh1.range('B5').value) 
    #float(input("Enter the width of the field: "))
    xMin=0;xMax=field_length
    yMin=0;yMax=field_width
    xDelta=xMax-xMin;yDelta=yMax-yMin; #rectangle dimensions
    areaTotal=xDelta*yDelta
    lambdaAP = 70e-6
    station_positions = []

    #Simulate Poisson point process
    num_stations = scipy.stats.poisson(lambdaAP*areaTotal).rvs()#Poisson number of points
    x = xDelta*scipy.stats.uniform.rvs(0,1,((num_stations,1)))+xMin#x coordinates of Poisson points
    y = yDelta*scipy.stats.uniform.rvs(0,1,((num_stations,1)))+yMin#y coordinates of Poisson points
    station_positions = list(zip(x,y))
    print("The number of the access points is: ")
    print(num_stations)
    #Plotting
    fig1 = plt.figure()
    plt.scatter(x,y, edgecolor='gold', facecolor='none', alpha=0.5,label="Access points"  )
    plt.xlabel("x [m]"); plt.ylabel("y [m]")

    number_users = int(sh1.range('B6').value) #int(input("Enter the number of users: "))
    dist_between_users = math.sqrt((field_length * field_width) / number_users)

    # Place stations at equal distance
    users_positions = []

    for i in range(number_users):
        x = random.uniform(0, field_length)
        y = random.uniform(0, field_width)
        users_positions.append((x, y))


    x_positions, y_positions = zip(*users_positions)
    plt.plot(x_positions, y_positions, '*', color='r', label='active users')
    for i, position in enumerate(users_positions):
        plt.annotate(i+1, position)
    
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    sh3.pictures.add(fig1, name='visualisation', update=True, left=sh1.range('A1').left, top=sh1.range('A1').top)



	#Largest-large-scale fading-based selection scheme
    L = num_stations
    K = number_users

    #in plaats van eerst gans de array te maken, maak een 1D array voor UE k en alle APs. Via een python functie:
    # 1) calculate total
    delta = 0.9
    Matrix_SUM_ALL_beta_lk = [0 for x in range(K)] 
    f = 1.9e3
    hAP = 15
    hU = 1.65
    L_f = -1*(46.3 + 33.9*math.log10(f) - 13.82*math.log10(hAP) - (1.1*math.log10(f)-0.7)*hU + (1.56*math.log10(f)-0.8))

    Matrix_lk = [[0 for x in range(K)] for y in range(L)] 
    for k in range(K):
        Matrix_beta_lk = [0 for x in range(L)] 
        for l in range(L):
            Matrix_beta_lk[l] = pow(10,(calculation_PLmk(l,k,users_positions, station_positions, L_f) + calculation_Fkl())/10)
            Matrix_SUM_ALL_beta_lk[k]  =Matrix_SUM_ALL_beta_lk[k]+ Matrix_beta_lk[l]
            
        # 2) make a tuple of it
        Matrix_beta_lk_tuple = [(i+1, value) for i, value in enumerate(Matrix_beta_lk)]
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
            #print(sum_AP)
    
    #Per user wordt het aantal AP die deze user served weergegeven
    M_k = [0 for x in range(K)]
    for k in range(K):
        sum = 0
        for l in range(L):
            sum = sum + Matrix_lk[l][k]
        M_k[k] = sum
    print("M_k: " + str(M_k))

    M_l = [0 for x in range(L)]
    for l in range(L):
        sum = 0
        for k in range(K):
            sum = sum + Matrix_lk[l][k]
        M_l[l] = sum
    print("M_l: " + str(M_l))
        
    print(type(Matrix_lk))
    end = time.time()
    print("Run time: " + str(end - start))
    print("DONE! :))")
    df = pd.DataFrame(Matrix_lk) 
    
    num_cols = len(df.columns)
    col_names = [f"UE {i+1}" for i in range(num_cols)]
    df.columns = col_names
    
    num_rows = len(df.index)
    row_names = [f"AP {i+1}" for i in range(num_rows)]
    df.index = row_names
    print(df)
    sh2.clear_contents()
    table = 'Selected APs'
    #sh2["A1"].options(pd.DataFrame, header=1, index=True, expand='table').value = df
    sh1.range("A11").value = "Done! You can find more information on the other sheets."
    if table in [table.name for table in sh2.tables]:
        sh2.tables[table].update(df)
    else:
        mytable = sh2.tables.add(source=sh2['A1'], name=table,table_style_name='TableStyleMedium12').update(df)
    
    T_coh = sh1.range('E4').value
    tauw_c = int(sh1.range('E5').value)
    zeta_dl= sh1.range('E6').value
    zeta_ul = sh1.range('E7').value
    tauw_tr =  K
    kappa = sh1.range('I11').value
    #U= tauw_c
    B= sh1.range('E9').value
    tauw_ul = (tauw_c-tauw_tr)*zeta_ul
    tauw_dl = (tauw_c-tauw_tr)*zeta_dl

    T_slot = T_coh #CHECK THIS!
    N = int(sh1.range('B7').value)
	
    C_ce = N*(2*tauw_tr -1)*K*L
    C_precRec = (3*pow(K,2)*L*N) + (pow(K,3)/3)
    C_LP = L*N*((2*K)-1)

    N_flops = (C_ce + C_LP + C_precRec)/(T_coh)
    flops_4nodes = sh1.range('I6').value
    N_4nodes = math.ceil(N_flops/flops_4nodes)

    P_idle_1x4nodes = sh1.range('I5').value
    P_idle = N_4nodes * P_idle_1x4nodes
    flopsPerWatt_4nodes = sh1.range('I7').value
    L_cu = N_4nodes * flopsPerWatt_4nodes
    P_fix = sh1.range('E10').value
    L_A = (df.sum(axis=1) > 0).sum()
    ################################################################ YOU StiLL HAVE to chANGe theSE!!!
    
    P_traffic = sh1.range('E12').value
    


    eta_AP = sh1.range('I10').value
    p_lk = sh1.range('E11').value
    
    P_tx =p_lk* L_A*((tauw_c-tauw_tr)/(tauw_c*eta_AP))

    P_FH_LI= L* P_fix
    P_AP = (22.6e-3 + N*(8.7e-3) + (2*kappa*N*0.1249) + (2*(1-kappa)*N*2.598962e-3))*L_A
    P_ce = C_ce*(B/(tauw_c*L_cu))
    P_precRec= C_precRec*(B/(tauw_c*L_cu))
    P_LP = C_LP*(B/(L_cu)) * (1-((tauw_tr)/tauw_c))
    N_mem_access = (N*tauw_tr) + (tauw_tr) + (N*tauw_tr*tauw_tr) + (2*L*N*K*K) + (K*K) + 2*((L*N*K)+((pow(K,2)-K)/(2))+K)
    P_MEM = (N_mem_access*5e-12)/(T_slot/2)
    P_CU = P_idle
    
    r_k = (1-(tauw_tr/tauw_c))*6.658211 #  -->with the ...log(1+SNR) = log(1+100)=6.65822 //20 dB = SNR
    P_rl = 0
    for l in range(L):
        P_rl = P_rl + (M_l[l]*r_k)
    P_FH_LD = B*(P_traffic*pow(10,-9))*P_rl
    
    #assuming the same rate for up en downlink = R
    
    P_cod = sh1.range('E13').value #W/Gbit/s
    P_dec = sh1.range('E14').value #W/Gbit/s
    P_cod_dec= B*((P_cod*pow(10,-9))+(P_dec*pow(10,-9)))*P_rl


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


