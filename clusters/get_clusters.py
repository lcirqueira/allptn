import os , time
import numpy as np
from scipy.optimize import curve_fit
from pyvmd import *
import atomsel
import macro_kv1
import macro_sitesKv
import gridData as gd
import sys
sys.path.append("/home/lcirqueira/Simulations/ionchannel/kv1/kv1.12/flooding/sevoflurane/analysis/allptn")
import allptn_definitions as ptndef
sys.path.append("../../")
import folderdefs as folddefs

CONC = folddefs.FOLDER_CONC

if 0:
    PATH = ptndef.AQUAFLD_PATH[CONC]
    NAME = ptndef.AQUAFLD_NAME[CONC]
else:
    PATH = "{}/lightdcd".format(ptndef.FLD_PATH[CONC])
    NAME = ptndef.LDCD_NAME[CONC]

FIRST, LAST = ptndef.DCD_NUM[CONC]
STEP = ptndef.STEP[CONC]

UNISEL = "segname SEV and name C2"

SEL = ( ("segname SEV and name C2 and (within 5 of protein) and (not (((x^2 + y^2) < 144) and z > -27 and z < 0 )) and (z > -25) and (z < 25)" , "tmptn"), 
        ("segname SEV and name C2 and (within 5 of protein) and ( ((x^2 + y^2) < 144) and z > -27 and z < 0 ) and (z > -25) and (z < 25)", "centralpore"),
        ("segname SEV and name C2 and (within 5 of protein) and not ((z > -25) and (z < 25))" , "wt"))[0:1]

TIME_PER_FRAME = 0.02 #ns


CUTOFF = 7

MAX_CLUSTER = 10

countdic = {}
allcountdic = {}
coordsdic = {}
for sel , selname in SEL:
    countdic[selname] = {n : [] for n in range(1 , MAX_CLUSTER + 1)}
    allcountdic[selname] = 0
    coordsdic[selname] = {n :  np.array([]).reshape(0,3) for n in range(1 , MAX_CLUSTER + 1)}



ref = System()
ref.load("{}/{}.0.psf".format(PATH, NAME))
ref.load("{}/{}.{}.dcd".format(PATH, NAME, FIRST), first=0 , last=0)
ref.wrap("protein")
ref.all.moveby(-ref.all.center)
refsel = ref.selectAtoms("backbone")


framesnum = 0
mol = System()
mol.load("{}/{}.0.psf".format(PATH, NAME))
for i in range(FIRST, LAST + 1):
    mol.load("{}/{}.{}.dcd".format(PATH, NAME, i), step=STEP)
    mol.wrap("protein")
    for frame in mol.trajectory:
        #mol.all.moveby(-mol.all.center)

        fitsel = mol.selectAtoms("backbone")
        fitmat = fitsel.fit(refsel)
        mol.all.move(fitmat)

        for sel , selname in SEL:
            ligsel = mol.selectAtoms(sel)
            maxlig = len(ligsel)


            resid_list = ligsel["residue"]
            cont_list = []
            for resid in resid_list:
                #nearsel = mol.selectAtoms("({0}) and within {1} of (residue {2})".format(sel , CUTOFF , resid))
                nearsel = mol.selectAtoms("({0}) and within {1} of ({2} and residue {3})".format(sel , CUTOFF , UNISEL , resid))
                temp_inter = list(set(nearsel["residue"]))
                cont_list.append(temp_inter)


            for resid in resid_list:
                flat_list = []
                for k, pair in enumerate(cont_list):
                    if resid in pair:
                        pop_item = cont_list.pop(k)
                        for item in pop_item:
                            flat_list.append(item)

                cont_list.append(list(set(flat_list)))

            flat_cont_list = [b for a in cont_list for b in a]
            if len(set(flat_cont_list)) != maxlig:
                print("error")


            for n in range(1 , MAX_CLUSTER + 1):
                countdic[selname][n].append(0)

            for itm in cont_list:
                countdic[selname][len(itm)][-1] += len(itm)
                allcountdic[selname] += len(itm)

                tmpcoordsel = mol.selectAtoms("{} and residue {}".format(UNISEL , " ".join(map(str ,itm))))
                coordsdic[selname][len(itm)] = np.append(coordsdic[selname][len(itm)] , tmpcoordsel.coords , axis=0)



        framesnum += 1

    mol.delFrame()

for sitename in countdic.keys():
    for num in countdic[sitename].keys():
        outarr = np.zeros((len(countdic[sitename][num]) , 2))
        outarr[:,0] = np.arange(outarr.shape[0]) * TIME_PER_FRAME
        outarr[:,1] = countdic[sitename][num]

        if set(countdic[sitename][num]) == set([0]):
            continue

        np.savetxt("ligtime.cluster.{}.{}.dat".format(sitename , num) , outarr , fmt="%.2f %.0f")


        occ_coords = coordsdic[selname][num]
        tempoccsys = System(atoms=occ_coords.shape[0])
        tempoccsys.dupFrame()
        tempoccsys.all.coords = occ_coords
        tempoccsys.all.write("pdb" , "coords.{}.{}.pdb".format(sitename , num))

        evaltcl("""volmap density [atomselect {0} all] -res 1.0 -o tempvolmap.dx""".format(tempoccsys.id))
        tmpdx = gd.Grid("tempvolmap.dx")
        tmpdx.grid *=  (1 / tmpdx.grid.sum())
        tmpdx.export("volmap.cluster.{}.{}.dx""".format(sitename , num))
        tempoccsys.delete()


    count_arr = np.zeros((MAX_CLUSTER , 2))
    count_arr[:,0] = np.arange(1 , MAX_CLUSTER + 1)
    for k in range(1 , MAX_CLUSTER + 1):
        count_arr[k - 1 , 1] = sum(countdic[sitename][k]) 

    np.savetxt("count.cluster.{}.dat""".format(sitename) , count_arr , fmt="%.0f")

    probs_arr = count_arr.copy()
    probs_arr[:,1] /= probs_arr[:,1].sum()

    np.savetxt("probs.cluster.{}.dat""".format(sitename) , probs_arr , fmt="%.0f %.4f")


print(time.process_time() / 60)

exit()
