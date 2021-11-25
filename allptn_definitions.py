import numpy as np

FLD_DENS = 150 #mM
REF_CONC = FLD_DENS

#using pdef eq_dcd
#EQ_DCD = {0 : 2 , 
          #1 : 15 ,
          #25 : 65 ,
          #50 : 75 ,
          #75 : 75 ,
          #100 : 85 ,
          #150 : 105 }

MOL_FRACTION = {150 : 0.29,
                100 : 0.21,
                 75 : 0.16,
                 50 : 0.11,
                 25 : 0.06,
                 1  : 0.0043} 

LIGNUM = {150 : 174,
          100 : 116,
           75 : 87,
           50 : 58,
           25 : 29,
           1  : 2} 


A3_TO_Lmol = 6.022e-4 ##binding constant convertion factor from Angstrom to (mol/L)^-1

FIRST = 5
STEP = 1
FRAMES_PER_DCD = 50

TIME_PER_FRAME = 0.02 #ns
EQ_START = 100 #ns
EQ_START_FRAME = int((EQ_START - FIRST) * (FRAMES_PER_DCD / STEP))


FLD_PATH = {150 : "/home/lcirqueira/Simulations/ionchannel/kv1/kv1.12/flooding/sevoflurane",
            100 : "/home/lcirqueira/Simulations/ionchannel/kv1/kv1.12/flooding/sevo100mM",
             75 : "/home/lcirqueira/Simulations/ionchannel/kv1/kv1.12/flooding/sevo75mM",
             50 : "/home/lcirqueira/Simulations/ionchannel/kv1/kv1.12/flooding/sevo50mM",
             25 : "/home/lcirqueira/Simulations/ionchannel/kv1/kv1.12/flooding/sevo25mM",
             1  : "/home/lcirqueira/Simulations/ionchannel/kv1/kv1.12/flooding/sevo1mM",
             0  : "/home/lcirqueira/Simulations/ionchannel/kv1/kv1.12/jhosoume",}

AQUAFLD_PATH = {150 : "",
            100 : "",
             75 : "",
             50 : "",
             25 : "",
             1  : "",
             0  : "",}

AQUAFLD_NAME = {150 : "",
            100 : "",
             75 : "",
             50 : "",
             25 : "",
             1  : "",
             0  : "",}

LDCD_NAME = {0 : "kv1.12.ldcd" , 
            1 : "kv1.12.sev1mM" ,
            25 : "kv1.12.sev25" ,
            50 : "kv1.12.sev50" ,
            75 : "kv1.12.sev75" ,
            100 : "kv1.12.sev100" ,
            150 : "kv1.12.protsev" }


DCD_NUM = {0 : (105, 304) , 
            1 : (105, 304) ,
            25 : (105, 304) ,
            50 : (105, 504) ,
            75 : (105, 304) ,
            100 : (105, 304) ,
            150 : (105, 504) }

STEP = {0 : 1 , 
        1 : 1 ,
        25 : 1 ,
        50 : 1 ,
        75 : 1 ,
        100 : 1 ,
        150 : 1 }


PSEUDO_RANGE = (0.05, 0.05 , 0.05)
pseudo_arr = np.arange(PSEUDO_RANGE[0], PSEUDO_RANGE[1] + PSEUDO_RANGE[2], PSEUDO_RANGE[2])
def calc_pseudoprobs(data, pseudo_factor):
    
    alph_size = np.trim_zeros(data[:,1] , 'b').shape[0]

    rawhist = data.copy()
    rawhist[:,1:] /= rawhist[:,1].sum()

    pseudohist = data.copy()
    pseudohist[:alph_size,1] = pseudo_factor / alph_size

    newhist = data.copy()
    newhist[:,1] = pseudohist[:,1] + (1 - pseudo_factor) * rawhist[:,1]

    return(newhist)


def getbindk(mat , dens):
    hist = mat.copy()

    trimhist = np.trim_zeros(hist[:,1], 'b')
    sevnum = trimhist.shape[0] - 1

    probs = np.empty((sevnum, sevnum))
    for i in range(sevnum):
        probs[:,i] = trimhist[:-1]

    common_matrix = np.empty((sevnum, sevnum))
    for j in range(sevnum):
        common_matrix[:,j] = np.power(dens, j+1) 

    diff_matrix = np.zeros((sevnum, sevnum))
    for k in range(1,sevnum):
        diff_matrix[k,k-1] = -1

    coefficients = common_matrix * (probs + diff_matrix)

    if trimhist.shape[0] > 1:
        values = -trimhist[:-1]
        values[0] += 1 

        solution = np.linalg.solve(coefficients , values)
        bindk_array = np.ones(solution.shape[0] + 1)
        bindk_array[1:] = solution

    else:
        bindk_array = np.array([1.0,0])

    return(bindk_array)

SITES = ("tmptn" , "centralpore" , "wt")    
