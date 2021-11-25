import os
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
OUT = "kv1.12"
AVG_OUT = "avg_{}".format(NAME)

ALL = "segname SEV and name C2"

#biggest bound selection
BIG_SEL = "segname SEV and name C2 and within 5 of protein" 

SEL = ( ("segname SEV and name C2 and (within 5 of protein) and (not (((x^2 + y^2) < 144) and z > -27 and z < 0 )) and (z > -25) and (z < 25)" , "tmptn"), 
        ("segname SEV and name C2 and (within 5 of protein) and ( ((x^2 + y^2) < 144) and z > -27 and z < 0 ) and (z > -25) and (z < 25)", "centralpore"),
        ("segname SEV and name C2 and (within 5 of protein) and not ((z > -25) and (z < 25))" , "wt"))


OUT_DIR = ("probs" , "count" , "ligcenter" , "ligtime" , "volmap")
for outdir in OUT_DIR:
    if not os.path.exists(outdir):
        os.mkdir(outdir)

##making selection list
sellist = []
outlist = []
for i in range(len(SEL)):
    sellist.append(SEL[i][0])
    outlist.append(SEL[i][1])


ref = System()
ref.load("{}/{}.0.psf".format(PATH, NAME))
ref.load("{}/{}.{}.dcd".format(PATH, NAME, FIRST), first=0 , last=0)
ref.wrap("protein")
ref.all.moveby(-ref.all.center)
refsel = ref.selectAtoms("backbone")


mol = System()
mol.load("{}/{}.0.psf".format(PATH, NAME))
all_coords = np.zeros((len(mol.all) , 3))
allsev = mol.selectAtoms(ALL)
sevnum = len(allsev)
minmax = np.zeros((2,3))

frames_num = 0
counter = np.zeros((len(sellist) , sevnum + 1 , 2))
counter[:,:,0] = np.arange(sevnum + 1)

histogram = np.zeros((len(sellist) , sevnum + 1 , 2))
histogram[:,:,0] = np.arange(sevnum + 1)

all_ligtime = []
allcount = counter[0].copy()
allhist = histogram[0].copy()

ligtime = []
ligcoords = []
occlist = []
for i in range(len(sellist)):
    ligtime.append([])
    ligcoords.append([])
    occlist.append([])

all_lig_coords = np.zeros((0,3))

for i in range(FIRST, LAST + 1):
    mol.load("{}/{}.{}.dcd".format(PATH, NAME, i), step=STEP)
    mol.wrap("protein")
    for frame in mol.trajectory:
        idxstring = ""
        first_sel = True

        fitsel = mol.selectAtoms("backbone")
        fitmat = fitsel.fit(refsel)
        mol.all.move(fitmat)

        all_coords += mol.all.coords

        for n, sel in enumerate(sellist):
            if first_sel:
                ligsel = mol.selectAtoms("({}) and name C2".format(sel))
            else:
                ligsel = mol.selectAtoms("({}) and name C2 and not (index {})".format(sel, idxstring))

            for idx in ligsel["index"]:
                idxstring += "{} ".format(idx)

            lignum = len(ligsel)


            if lignum:
                ligcoords[n].append(ligsel.coords)
                occlist[n] += (np.ones(lignum) * lignum).tolist() 
                first_sel = False

            ligtime[n].append(lignum)
            counter[n, lignum, 1] += 1

        

        allsitesel = mol.selectAtoms(BIG_SEL)
        all_lig_coords = np.append(allsitesel.coords , all_lig_coords , axis=0)
        all_num = len(allsitesel)
        all_ligtime.append(all_num)
        allcount[all_num , 1] += 1

        frames_num += 1

    mol.delFrame()

mol.dupFrame(-1)
mol.all.coords = all_coords / frames_num
mol.selectAtoms("protein and noh").write("psf", "{}/{}.psf".format(OUT_DIR[2],AVG_OUT))
mol.selectAtoms("protein and noh").write("pdb", "{}/{}.pdb".format(OUT_DIR[2],AVG_OUT))


histogram[:,:,1] = counter[:,:,1] / frames_num
allhist[:,1] = allcount[:,1] / frames_num


with open("sites_lignumber.dat" , "w") as mlig:
    mlig.write("##individual subunits mean numbers:\n")
    for j in range(len(outlist)):
        np.savetxt("{}/hist.{}.dat".format(OUT_DIR[0] , outlist[j]) , histogram[j] , fmt="%d %.6f")
        np.savetxt("{}/count.{}.dat".format(OUT_DIR[1] , outlist[j]) , counter[j] , fmt="%d %d")
        np.savetxt("{}/ligtime.{}.dat".format(OUT_DIR[3] , outlist[j]) , ligtime[j] , fmt="%d")


        mlig.write("{} : {}\n".format(outlist[j] , np.sum(np.prod(histogram[j], axis = 1))))

    meantot = allhist.prod(axis=1).sum()

    mlig.write("""
total ligand number: {}
""".format(meantot))
    
#create pdb with bound ligands
for b in range(len(ligcoords)):
    if len(ligcoords[b]):
        temparr = np.concatenate(ligcoords[b])
        tempsys = System(atoms=temparr.shape[0])
        tempsys.dupFrame()
        tempsys.all.coords = temparr
        tempsys.all["occupancy"] = occlist[b]
    else:
        tempsys = System(atoms=1)
        tempsys.dupFrame()
    tempsys.all.write("pdb" , "{}/coords.{}.pdb".format(OUT_DIR[2], outlist[b]))

    evaltcl("""volmap density [atomselect {0} all] -res 1.0 -o tempvolmap.dx""".format(tempsys.id))
    tmpdx = gd.Grid("tempvolmap.dx")
    tmpdx.grid *= (1/frames_num)
    tmpdx.export("{}/volmap-{}.dx""".format(OUT_DIR[4] , outlist[b]))
    tempsys.delete()


np.savetxt("{}/hist.all.dat".format(OUT_DIR[0]) , allhist , fmt="%d %.6f")
np.savetxt("{}/count.all.dat".format(OUT_DIR[1]) , allcount , fmt="%d %d")
np.savetxt("{}/ligtime.all.dat".format(OUT_DIR[3]) , all_ligtime , fmt="%d")

tempallsys = System(atoms=all_lig_coords.shape[0])
tempallsys.dupFrame()
tempallsys.all.coords = all_lig_coords
tempallsys.all.write("pdb" , "{}/coords.all.pdb".format(OUT_DIR[2]))

evaltcl("""volmap density [atomselect {0} all] -res 1.0 -o tempvolmap.dx""".format(tempallsys.id))
tmpdx = gd.Grid("tempvolmap.dx")
tmpdx.grid *= (1/frames_num)
tmpdx.export("{}/volmap-all.dx""".format(OUT_DIR[4]))
tempallsys.delete()
    
exit()
