import parmed as pmd
from paprika import build
from paprika import restraints
from paprika import amber
from paprika import utils
import numpy as np
import os
import shutil
import re

def test_amber_single_window_min():
    # Align PDB to Z-axis
    inputpdb = pmd.load_file('cb6-but/cb6-but-notcentered.pdb')
    
    # Distance restraint
    rest1 = restraints.DAT_restraint()
    rest1.continuous_apr = True
    rest1.amber_index = True
    rest1.topology = inputpdb
    rest1.mask1 = ':CB6@O'
    rest1.mask2 = ':BUT@C1'
    rest1.attach['target'] = 4.5
    rest1.attach['fraction_list'] = [0.00, 0.04, 0.181, 0.496, 1.000]
    rest1.attach['fc_final'] = 5.0
    rest1.pull['fc'] = rest1.attach['fc_final']
    rest1.pull['target_initial'] = rest1.attach['target']
    rest1.pull['target_final'] = 18.5
    rest1.pull['num_windows'] = 19
    rest1.initialize()
    
    # Angle restraint
    rest2 = restraints.DAT_restraint()
    rest2.continuous_apr = True
    rest2.amber_index = True
    rest2.topology = inputpdb
    rest2.mask1 = ':CB6@O1'
    rest2.mask2 = ':CB6@O'
    rest2.mask3 = ':BUT@C1'
    rest2.attach['target'] = 8.0
    rest2.attach['fraction_list'] = [0.00, 0.04, 0.181, 0.496, 1.000]
    rest2.attach['fc_final'] = 50.0
    rest2.pull['fc'] = rest2.attach['fc_final']
    rest2.pull['target_initial'] = rest2.attach['target']
    rest2.pull['target_final'] = rest2.attach['target']
    rest2.pull['num_windows'] = 19
    rest2.initialize()
    
    # Dihedral restraint
    rest3 = restraints.DAT_restraint()
    rest3.continuous_apr = True
    rest3.amber_index = True
    rest3.topology = inputpdb
    rest3.mask1 = ':CB6@O11'
    rest3.mask2 = ':CB6@O1'
    rest3.mask3 = ':CB6@O'
    rest3.mask4 = ':BUT@C1'
    rest3.attach['target'] = -60.0
    rest3.attach['fraction_list'] = [0.00, 0.04, 0.181, 0.496, 1.000]
    rest3.attach['fc_final'] = 50.0
    rest3.pull['fc'] = rest3.attach['fc_final']
    rest3.pull['target_initial'] = rest3.attach['target']
    rest3.pull['target_final'] = rest3.attach['target']
    rest3.pull['num_windows'] = 19
    rest3.initialize()

    # Create working directory
    if os.path.exists('amber_test'):
        shutil.rmtree('amber_test')
    os.makedirs('amber_test')
    path = './amber_test/'

    # Write restraints file for amber
    with open(path+'restraints.in', 'w') as f:
        for rest in [rest1,rest2,rest3]:
            # Testing just window p005
            f.write(restraints.amber_restraint_line(rest,'pull',5))

    # Copy build files for tleap
    files = 'cb6.mol2 cb6.frcmod but.mol2 but.frcmod cb6-but-notcentered.pdb'.split()
    for file in files:
        shutil.copy('cb6-but/'+file,path+file)

    # Build prmtop/inpcrd
    build.basic_tleap('cb6-but/tleap_gb.in', directory=path, pdbfile='cb6-but-notcentered.pdb', saveprefix='vac')

    # Create Simulation
    gbsim = amber.Simulation()
    gbsim.path = path
    gbsim.executable = 'sander'
    gbsim.topology = 'vac.prmtop'
    gbsim.prefix = 'minimize'
    gbsim.inpcrd = 'vac.rst7'
    gbsim.config_gb_min()
    gbsim.cntrl['maxcyc'] = 1
    gbsim.cntrl['ncyc'] = 1 
    gbsim.run()

    # Collect values
    test_values = []
    with open(path+'minimize.out', 'r') as f:
        filelines = f.readlines()
        for i,line in enumerate(filelines):
            if re.search('^ BOND ', line):
                cols = line.split()
                test_values.append(float(cols[2])) # BOND
                test_values.append(float(cols[5])) # ANGLE
                test_values.append(float(cols[8])) # DIHED
                cols = filelines[i+1].split()
                test_values.append(float(cols[2])) # VDWAALS
                test_values.append(float(cols[5])) # EEL
                test_values.append(float(cols[8])) # EGB
                cols = filelines[i+2].split()
                test_values.append(float(cols[3])) # 1-4 VDW
                test_values.append(float(cols[7])) # 1-4 EEL
                test_values.append(float(cols[10])) # RESTRAINT
                cols = filelines[i+3].split()
                test_values.append(float(cols[2])) # EAMBER
                cols = filelines[i+4].split()
                test_values.append(float(cols[4])) # Restraint: Bond
                test_values.append(float(cols[7])) # Restraint: Angle
                test_values.append(float(cols[10])) # Restraint: Torsion
                break

    # A bit ugly, but we're here
    nptest_values = np.asarray(test_values)
    # Reference             BOND      ANGLE    DIHED    VDWAALS     EEL        EGB    1-4 VDW    1-4 EEL  RESTRAINT   EAMBER    Bond    Angle  Torsion
    ref_values = np.array([11.4252, 109.4691, 52.9473, -67.7555, 1326.4786, -123.9177, 5.8587, -2127.9397, 90.8046, -813.4340, 76.073, 14.724, 0.008])

    assert np.allclose(nptest_values, ref_values)

    shutil.rmtree('amber_test')


#test_amber_single_window()

