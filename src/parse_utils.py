"""
A file with data manipulation utils. Used for auto-reading and transposing directories with output files to csv format.
To use, writh the command
        python parse_utils.py TARGET_DIRECTORY
It will parse the outfiles in a directory to a results.csv file in the directory.
NOTE: when running on different directories (top N or top Cr, 111 or 200 planes...) care should be taken as some of the calculation parameters should be changed.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, fsolve
from scipy.integrate import quad

from utils import *

parser = argparse.ArgumentParser(description='Running parameters')
parser.add_argument('working_path', type=str, help='Path containing all files for the computation. Including cif file for structure.')
parser.add_argument('--correct_distances', type=bool, default=True, help='Correct distances to uniform scale')
parser.add_argument('--correct_energies', type=bool, default=True, help='Correct energies to ads energies in eV')

args = parser.parse_args()

# setting up plotting parameters
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# base energies for structure and molecules
base_energies = {
        'mol': -31.73448235,
        'atom': -12.94925931,
        'Cr': -247.5732555404,
        'N':  -27.6234602498,
        # top N
        #'111': {'None': -9951.5505324365, '0': -9703.2078759678, '1': -9923.3499238508},
        # top Cr
        '111': {'None': -9951.1370129207, '0': -9703.3434081283},
        '200': {'None': -9952.144536, '0': -9704.00001, '1': -9923.912495}
    }
    
abc_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E'
}

def _name_to_dict(name):
    d = {'plane': None, 'defect': None, 'specie': None, 'pos': None, 'dist': None}
    if "atom" in name.lower():
        d['specie'] = 'atom'
    if 'mol' in name.lower() or 'molecule' in name.lower():
        d['specie'] = 'mol'
    
    d['plane'] = name[:3]
    
    if 'defect' in name.lower():
      idx = name.index('t')
      d['defect'] = int(name[idx + 2])
    
    if 'pos' in name.lower():
      idx = name.index('s')
      d['pos'] = int(name[idx + 2])

    return d
    
def name_to_dict(name):
    d = {'plane': '111', 'defect': None, 'specie': None, 'pos': None, 'dist': None}
    v = name.split("_")
    get_in = lambda in_name: v[v.index(in_name) + 1] if in_name in v else None
    d['specie'] = get_in('specie')
    d['pos'] = int(get_in('pos'))
    d['dist'] = get_in('dist')
    return d
    
def get_z_from_outdict(d):
  vec = d['cell_params'][-1]
  z_list = []
  for pos in d['atom_pos']:
    z_list.append(np.dot(np.array(pos), np.array(vec)))
  return z_list

def get_base_atom(atom, atoms_list, z_list, find='max'):
  if find == 'max':
      highest_z = 0
      for a, z in zip(atoms_list, z_list):
          if a == atom and z > highest_z:
            highest_z = z
  if find == 'min':
      highest_z = 100
      for a, z in zip(atoms_list, z_list):
          if a == atom and z < highest_z:
            highest_z = z
  return highest_z

def morse_potential(r, De, a, re, c):
    return De * np.square((1 - np.exp(-a * (r - re)))) + c

def morse_potential_err(x, t, y):
    return morse_potential(t, x[0], x[1], x[2], x[3]) - y
    
def rejection_potential(r, a, c):
    return a / np.square(r - c)

def rejection_potential_err(x, t, y):
    return rejection_potential(t, x[0], x[1]) - y

def add_plot(x, y, fmt, label, fit_to=None):
    if len(x) == 0 or len(y) == 0:
        return 
    color, marker = fmt
    
    fig = plt.gcf()
    ax = plt.gca() 
    ax.scatter(x, y, c=color, marker=marker, label=label)
    ax.legend(loc='upper right')
    
    # fitting data to standard potentials
    if not fit_to == None:
        x_range = np.linspace(min(x) - 0.1, max(x), 100)
        if fit_to == 'morse_potential':
            res = least_squares(morse_potential_err, [10, 1, 1, -10], args=(x, y))
            fitted_pot = morse_potential(x_range, *res.x)
        if fit_to == 'rejection_potential':
            res = least_squares(rejection_potential_err, [1, 100], args=(x, y))
            fitted_pot = rejection_potential(x_range, *res.x)
        ax.plot(x_range, fitted_pot, color + "-")
        
    x_min = min(0, 1.05 * min(x)); x_max = 1.05 * max(x)
    y_min = min(0.95 * min(y), 1.05 * min(y)); y_max = min(10, 1.05 * max(y))
    
    if x_min < plt.xlim()[0]:
      plt.xlim(left=x_min)
    if x_max > plt.xlim()[1]:
      plt.xlim(right=x_max)
    if y_min < plt.ylim()[0]:
      plt.ylim(bottom=y_min)
    plt.ylim(top=y_max)
    
    plt.xlabel("Distance [A]")
    plt.ylabel("Ads Energy [eV]")
    
def get_distance(d, correct_dists, base='max'):
    '''Returns distance between hydrogen and the surface. correction to get the distance to the highest chromium atom'''
    z_list = get_z_from_outdict(d)
    Hidx = [i for i, v in enumerate(d['atom_species']) if v == "H"]
    Hzeds = [z_list[i] for i in Hidx]
    Hidx = Hidx[Hzeds.index(min(Hzeds))]
    if correct_dists:
        base_z = get_base_atom("N", d['atom_species'], z_list, base)
        if base == 'max':
            dist = z_list[Hidx] - base_z
        elif base == 'min':
            dist = base_z - z_list[Hidx]
        # correction in case that distance is greater than lattice params
        if base == 'max':
            dist_list = z_list[Hidx] - np.array(z_list)
        elif base == 'min':
            dist_list = np.array(z_list) - z_list[Hidx]
        add_counter = 0
        while True:
          if all([x <= 0 for x in dist_list]):
            add_counter += 1
            dist_list += np.linalg.norm(d['cell_params'][-1])
          else:
            break
        return dist + add_counter * np.linalg.norm(d['cell_params'][-1])
    else:
        return z_list[Hidx]

def get_energy(d, correct_energy, plane, specie, defect):
    '''returns total energy in eV. Correction to get ads energy'''
    eng = d['total_energy']
    if correct_energy:
        return (eng - base_energies[plane][str(defect)]) * 13.6 - base_energies[specie]
    else:
        return eng

def parse_directory(work_dir, correct_distance, correct_energy):
    df = pd.DataFrame()
    for filename in os.listdir(work_dir):
        if filename.endswith('.out'):
            print(filename)
            in_d = name_to_dict(filename)
            path = os.path.join(work_dir,filename)
            if filename.endswith('.out'):
                d = ReadOutput(path)
                in_d = name_to_dict(filename[:-4])
                if in_d == False:
                  continue  
    
                in_d['dist'] = get_distance(d, correct_distance)
                in_d['total_e'] = get_energy(d, correct_energy, in_d['plane'], in_d['specie'], in_d['defect'])
            in_d['filename'] = filename
            df = df.append(in_d, ignore_index=True)
    return df
    
def make_plot_data(df, specie, defect, pos, plane, verbose=1):
    if not defect == None:
        if not pos == None:
            data = df[(df['defect'] == defect) & (df['pos'] == pos) & (df['plane'] == plane) & (df['specie'] == specie)]
        else:
            data = df[(df['defect'] == defect) & (df['plane'] == plane) & (df['specie'] == specie)]
    else:
        if not pos == None:
            data = df[(df['pos'] == pos) & (df['plane'] == plane) & (df['specie'] == specie)]
        else:
            data = df[(df['plane'] == plane) & (df['specie'] == specie)]
    data = data.sort_values(by='dist')
    if verbose > 0:
        if not defect == None:
            print("\n" + "*" * 10 + " " * 10 + f"{plane} {defect} {specie} {pos}" + " " * 10 + "*" * 10 + "\n")
        else:
            print("\n" + "*" * 10 + " " * 10 + f"{plane} {specie} {pos}" + " " * 10 + "*" * 10 + "\n")
        print(data)
    return data  

def make_potential_plot(df, specie, defect, pos, plane, fmt):
    data = make_plot_data(df, specie, defect, pos, plane)
    if specie == 'mol':
        add_plot(data['dist'].values, data['total_e'].values, fmt, f"Position {abc_dict[pos]}", fit_to='rejection_potential')
    if specie == 'atom':
        add_plot(data['dist'].values, data['total_e'].values, fmt, f"Position {abc_dict[pos]}", fit_to='morse_potential')

def make_value_lists(df):
    defects = list(set(df['defect'].values))
    if defects == []:
        defects = [None]
    planes = list(set(df['plane'].values))
    positions = list(set(df['pos'].values))
    species = list(set(df['specie'].values))
    return defects, planes, positions, species   
 
def make_potential_plots(df, working_path, fmts=['ko', 'bs', 'gD', 'rv', 'm^']):
    defects, planes, positions, species = make_value_lists(df)
    
    plot_dir = os.path.join(args.working_path, "Plots")
    if not os.path.isdir(plot_dir):
        os.mkdir(os.path.join(plot_dir))
    for defect in defects:
        for plane in planes:
              for specie in species:
                  fig, ax = plt.subplots()
                  for fmt, pos in zip(fmts, positions):
                      make_potential_plot(df, specie, defect, pos, plane, fmt)
                  if not defect == None:
                      plt.savefig(os.path.join(plot_dir, 'plane_%s_defect_%s_specie_%s.png' % (plane, defect, specie)))
                  else:
                      plt.savefig(os.path.join(plot_dir, 'plane_%s_specie_%s.png' % (plane, specie)))

def make_dissociation_plots(df, working_path, fmts=['ko', 'bs', 'gD', 'rv', 'm^']):
    defects, planes, positions, species = make_value_lists(df)
    plot_dir = os.path.join(args.working_path, "Plots")
    if not os.path.isdir(plot_dir):
        os.mkdir(os.path.join(plot_dir))
    for defect in defects:
        for plane in planes:
            fig, ax = plt.subplots()
            # finding best position
            data = make_plot_data(df, 'atom', defect, None, plane)
            best_pos = data[(data['total_e'] == min(data['total_e'].values))]['pos'].values[0]
            
            # plotting molecule graphs
            for fmt, pos in zip(fmts, positions):
                data = make_plot_data(df, 'mol', defect, pos, plane)
                add_plot(data['dist'].values, data['total_e'].values, fmt, f"Position {abc_dict[pos]}", fit_to='rejection_potential')
            
            # plotting dissociated specie
            data = make_plot_data(df, 'atom', defect, best_pos, plane)
            energies = (np.array(data['total_e'].values) + base_energies['atom']) * 2 - base_energies['mol'] # correcting energy for dissociated specie
            add_plot(data['dist'].values, energies, fmts[-1], "Dissociated", fit_to='morse_potential')
            
            # saveing plots
            if not defect == None:
                plt.savefig(os.path.join(plot_dir, 'plane_%s_defect_%s_dissociated.png' % (plane, defect)))
            else:
                plt.savefig(os.path.join(plot_dir, 'plane_%s_dissociated.png' % (plane)))
                
def calc_barrier_energies(df, working_path):
    defects, planes, positions, species = make_value_lists(df)
    barrier_energies_df = pd.DataFrame()
    for defect in defects:
        for plane in planes:
            data = make_plot_data(df, 'atom', defect, None, plane)
        
            # getting dissociated potential func
            best_pos = data[(data['total_e'] == min(data['total_e'].values))]['pos'].values[0]
            diss_data = make_plot_data(df, 'atom', defect, best_pos, plane)
            diss_energies = (np.array(diss_data['total_e'].values) + base_energies['atom']) * 2 - base_energies['mol'] # correcting energy for dissociated specie
            dis_res = least_squares(morse_potential_err, [10, 1, 1, -10], args=(diss_data['dist'].values, diss_energies))
            
            # getting minimal barrier energie
            barrier_e = 1000
            for pos in positions:
                # getting mol funcs
                mol_data = make_plot_data(df, 'mol', defect, pos, plane)
                mol_res = least_squares(rejection_potential_err, [1, 100], args=(mol_data['dist'].values, mol_data['total_e'].values))
                # finding barrier energy
                solve_func = lambda x: morse_potential(x, *dis_res.x) - rejection_potential(x, *mol_res.x)
                e = morse_potential(fsolve(solve_func, 1)[0], *dis_res.x)
                if e < barrier_e:
                    barrier_e = e
            kb = 8.617333262e-5
            T = 298
            m = 1.6735575e-27 * 2
            mb_distribution = lambda x: np.sqrt(1 / (x * np.pi * kb * T)) * np.exp(-x / (kb * T))
            frac = quad(mb_distribution, barrier_e, np.inf)[0]
            str_defect = str(int(defect)) if not defect == None else str(defect)
            slab_e = (base_energies[str(plane)][str_defect] + base_energies['Cr' if defect == 0 else 'N'] - base_energies[str(plane)]["None"]) * 13.60566
            d = {'plane': plane, 'defect': defect, 'barrier (eV)': barrier_e, 'fraction': frac, 'slab energy': slab_e}
            barrier_energies_df = barrier_energies_df.append(d, ignore_index=True)
    barrier_energies_df.to_csv(os.path.join(working_path, 'barrier_energies.csv')) 
    print(barrier_energies_df)
            
def main():
    print("parsing directory...")
    df = parse_directory(args.working_path, args.correct_distances, args.correct_energies)
    df.to_csv(os.path.join(args.working_path, 'results.csv'))
    print(df)

if __name__ == '__main__':
    main()