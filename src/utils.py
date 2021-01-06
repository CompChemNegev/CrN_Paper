from pymatgen.core import Structure, Molecule
from pymatgen.core import Element, Lattice
from pymatgen.core import surface
from pymatgen.analysis import local_env, adsorption
import warnings
import numpy as np
import pandas as pd
from copy import copy, deepcopy
import re
import os

# Quantum ESPRESSO file generation and handling
def gen_prefix():
    with open('/home/shaharpit/utils/job.count', "r") as f:
        prefix = int(f.read())
    with open('/home/shaharpit/utils/job.count', "w") as f:
        f.write(str(prefix + 1))
    return prefix 

# Quantum ESPRESSO file generation and handling
def _correct_keywords_dict(struct: Structure, Dict)->dict:
    '''method to autocorrect and check input keywords dict. Returns corrected dict.'''
    if not 'CONTROL' in Dict.keys():
        Dict['CONTROL'] = dict()
    Dict['CONTROL']['prefix'] = str(gen_prefix())
    if not 'ELECTRONS' in Dict.keys():
        Dict['ELECTRONS'] = dict()
    if not 'SYSTEM' in Dict.keys():
        Dict['SYSTEM'] = dict()
    if not 'ibrav' in Dict['SYSTEM'].keys():
        Dict['SYSTEM']['ibrav'] = 0
        Dict['CELL_PARAMETERS'] = struct.lattice.matrix
    else:
        Dict['CELL_PARAMETERS'] = []
    Dict['SYSTEM']['nat'] = len(struct.species)
    Dict['SYSTEM']['ntyp'] = len(struct.types_of_specie)
    if not 'ATOMIC_SPECIES' in Dict.keys():
        raise ValueError('keywords_dict must contain an \'ATOMIC_SPECIES\' entry as a dict with species (keys) and potentials (values)')
    if not 'IONS' in Dict.keys():
        Dict['IONS'] = dict()
    if not 'CELL' in Dict.keys():
        Dict['CELL'] = dict()
    if not 'K_POINTS' in Dict.keys():
        raise ValueError('keywords_dict must contain an \'K_POINTS\' entry as a dict with \'type\' and \'vec\' keys')
    return Dict
    # TODO: add the rest of the parameters: OCUUPATIONS, CONSTRAINTS, ATOMIC_FORCES

def WriteInput(struct: Structure, filename, keywords_dict: dict):
    '''method to generate input for QE calculation'''
    # TODO: add particular explanation on keywords_dict
    _keywords_dict = deepcopy(keywords_dict)
    _keywords_dict = _correct_keywords_dict(struct, _keywords_dict)
    with open(filename, "w") as f:
        for key in ['CONTROL', 'SYSTEM', 'ELECTRONS', 'IONS', 'CELL']:
            f.write("&" + key + "\n")
            for k, v in _keywords_dict[key].items():
                if type(v) is str:
                    if not v == '':
                        s = "\t" + k + "=\'" + v + '\',\n'
                    else:
                        s = "\t" + k + ',\n'
                elif type(v) is bool:
                    if v:
                        s = "\t" + k + "=.TRUE,\n"
                    else:
                        s = "\t" + k + "=.FALSE,\n"
                else:
                    s = "\t" + k + "=" + str(v) + ',\n'
                f.write(s)
            f.write("/\n\n")
        f.write("ATOMIC_SPECIES\n")
        for key, value in _keywords_dict["ATOMIC_SPECIES"].items():
            mass = Element(re.findall(r'[A-Za-z]+', key)[0]).atomic_mass.real
            f.write(key + " " + str(mass) + " " + value + "\n")
        f.write("\nATOMIC_POSITIONS crystal\n")
        strs = []
        for site in struct.sites:
            strs.append(site.specie.symbol + " " + str(round(site.a, 4)) + " " + str(round(site.b, 4)) + " " + str(round(site.c, 4)))
        if 'ATOMIC_POSITIONS' in _keywords_dict:
            for idx, vec in enumerate(keywords_dict['ATOMIC_POSITIONS']):
                for v in vec:
                    strs[idx] = strs[idx] + " " + str(int(v))
        for string in strs:
            f.write(string + "\n")
        f.write(f"\nK_POINTS {keywords_dict['K_POINTS']['type']}\n")
        s = ""
        for v in _keywords_dict['K_POINTS']['vec']:
            s = s + str(v) + " "
        f.write("\t" + s + "\n")
        if not _keywords_dict['CELL_PARAMETERS'] == []:
          f.write("\nCELL_PARAMETERS {angstrom}")
          for vec in _keywords_dict['CELL_PARAMETERS']:
              s = ""
              for v in vec:
                  s = s + str(round(v, 4)) + " "
              f.write("\n\t" + s)
    # TODO: add the rest of the parameters: OCUUPATIONS, CONSTRAINTS, ATOMIC_FORCES

def RunInput(in_filename, out_filename, core_num, executable_path):
    os.run(f'mpirun -np {n_cores} {executable_path} < {in_filename} > {out_filename}')

def ReadOutput(filename)->dict:
    '''method to read output file of QE calculation, currently outputs only final total energy'''
    # TODO: make method more general and read more then energy - ESSPECIALY results from NEB calc
    Dict = dict()
    with open(filename, "r") as f:
        FinalCoordsBlock = False
        CartesianBlock = False
        CellParamsBlock = False
        AtomPosBlock = False
        CoordsAreCartesian = False
        Dict['atom_species'] = []
        Dict['atom_pos'] = []
        Dict['cell_params'] = []
        Dict['total_energy'] = None
        for line in f.readlines():
            if "!" in line:
                # take the last "total energy line"
                Dict['total_energy'] = float(re.findall(r'[\d|.|,|-]+', line)[0])
            if FinalCoordsBlock is True:
                if len(line) > 1 and not 'End' in line:
                    specie = re.findall(r'[A-Za-z]+', line)[0]
                    coords = [float(c) for c in re.findall(r'[\d|.|,|-]+', line)[:3]]
                    Dict['atom_species'].append(specie)
                    Dict['atom_pos'].append(coords)
                else:
                    FinalCoordsBlock = False
            if 'celldm(1)' in line:
                param_a = float(re.findall(r'[\d|.|,|-]+', line)[1]) * 0.529177 # correction (for some reason the a parameter is scaled) TODO: FIND OUT WHY
            if CellParamsBlock is True:
                if len(line) > 1:
                    vec = [float(c) * param_a for c in re.findall(r'[\d|.|,|-]+', line)[1:]]
                    Dict['cell_params'].append(vec)
                else:
                    CellParamsBlock = False
            if CartesianBlock:
                if len(line) > 1:
                    if not 'site' in line:
                        vec = [w for w in line.split() if not w == '']
                        Dict['atom_species'].append(vec[1])
                        Dict['atom_pos'].append([float(c) for c in vec[6:9]])
                else:
                    if blank_line_counter > 0:
                        blank_line_counter -= 1
                    else:
                        CartesianBlock = False
            if "crystal axes" in line:
              CellParamsBlock = True
            if 'ATOMIC_POSITIONS' in line:
                FinalCoordsBlock = True
                CoordsAreCartesian = False
                Dict['atom_species'] = []
                Dict['atom_pos'] = []
            if "Cartesian axes" in line:
                CartesianBlock = True
                CoordsAreCartesian = True
                Dict['atom_species'] = []
                Dict['atom_pos'] = []
                blank_line_counter = 1
        if CoordsAreCartesian:
            inverse_mat = np.linalg.inv(np.array(Dict['cell_params']))
            Dict['atom_pos'] = [np.dot(inverse_mat, np.array(vec)) * param_a for vec in Dict['atom_pos']]
        return Dict
        
def OutToCif(outfile, cif_file='default'):
    d = ReadOutput(outfile)
    struct = Structure(d['cell_params'], d['atom_species'], d['atom_pos'])
    if cif_file == 'default':
        cif_file = outfile[:-3] + 'cif'
    struct.to('cif', cif_file) 
    
def ReadInput(filename):
    Dict = {'atom_species': [], 'atom_pos': [], 'cell_params': []}
    with open(filename, "r") as f:
        AtomPosBlock = False
        CellParamsBlock = False
        for line in f.readlines():
            if "ATOMIC_POSITIONS" in line:
                AtomPosBlock = True
                continue
            if "CELL_PARAMETERS" in line:
                CellParamsBlock = True
                continue
            if AtomPosBlock and len(re.findall(r'[A-Za-z]+', line)) == 0:
                AtomPosBlock = False
                continue
            if CellParamsBlock and len(line) == 0:
                CellParamsBlock = False
                continue                
            if AtomPosBlock:
                specie = re.findall(r'[A-Za-z]+', line)[0]
                coords = [float(c) for c in re.findall(r'[\d|.|,|-]+', line)[:3]]
                Dict['atom_species'].append(specie)
                Dict['atom_pos'].append(coords)
                continue
            if CellParamsBlock:
                 vec = [float(c) for c in re.findall(r'[\d|.|,|-]+', line)[:3]]
                 Dict['cell_params'].append(vec)
                 continue
    return Dict
    
def InToCif(infile, cif_file='default'):
    d = ReadInput(infile)
    struct = Structure(d['cell_params'], d['atom_species'], d['atom_pos'])
    if cif_file == 'default':
        cif_file = infile[:-2] + 'cif'
    struct.to('cif', cif_file) 

def InFromOutfile(infile, outfile, keywords_dict):
    '''Write input file based on geometry in outfile. using keywords dictionary.
    ARGS:
      - infile: path to input file
      - outfile: path to output file
      -keywords_dict: keywords dictionary for input'''
    d = ReadOutput(outfile)
    struct = Structure(d['cell_params'], d['atom_species'], d['atom_pos'])
    WriteInput(struct, infile, keywords_dict)

# surface generation and handling
def generate_slabs(struct, slab_size, vacuum_size, max_idx=3):
    # TODO: predict optimal slab size and vacuum size for struct (?)
    miller_idxs = surface.get_symmetrically_distinct_miller_indices(struct, max_idx)
    slabs = []
    for miller_idx in miller_idxs:
        slabgen = surface.SlabGenerator(struct, miller_idx, slab_size, vacuum_size, in_unit_planes=True)
        for slab in slabgen.get_slabs():
            slabs.append(slab)
    return slabs

def get_neighbor_idxs(slab, n, env_strat):
    info = env_strat.get_nn_info(slab, n)
    idxs = None
    if type(info) is list:
        idxs = set()
        for d in info:
            idxs.add(d['site_index'])
    elif type(info) is tuple:
        idxs = []
        for t in info:
            try:
                idx = slab.sites.index(t[0])
                idxs.append(idx)
            finally:
                pass
    return list(idxs)

def get_top_layer(slab, tol, env_strat=local_env.BrunnerNN_real(), method='max'):
    # find first atom in surface
    # the first atom is the highest one in the unit cell
    hights = [np.abs(site.c * slab.lattice.c) for site in slab.sites]
    if method == 'max':
        first_idx = hights.index(max(hights))
    elif method == 'min':
        first_idx = hights.index(min(hights))
    else:
        raise ValueError("method must be max or min")
    final_idxs = set([first_idx])
    new_idxs= set([first_idx])
    while True:
        idxs = set()
        for i in new_idxs:
            neighbors = set()
            if not i in final_idxs or len(final_idxs) == 1:
                for n in get_neighbor_idxs(slab, i, env_strat):
                    if not n in final_idxs and not n in new_idxs:
                        neighbors.add(n)
                for idx in neighbors:
                    dist = np.abs(hights[first_idx] - np.abs(slab.sites[idx].c * slab.lattice.c))
                    if dist < float(slab.sites[idx].specie.atomic_radius) + tol:
                        idxs.add(idx)
        final_idxs = final_idxs.union(new_idxs)
        if idxs == set():
            break
        new_idxs = copy(list(idxs))
    final_idxs = final_idxs.union(new_idxs)
    return list(final_idxs)


def get_layers(slab, env_strat=local_env.BrunnerNN_real(), tol=0.5, method='max'):
    '''returns the indices of the atoms in layers. tol is used to determine weather two atoms are in the same layer'''
    top_idxs = get_top_layer(slab, tol, env_strat, method)
    layers = [top_idxs]
    while True:
        sorted_idxs = [idx for idxs in layers for idx in idxs]
        new_layer = []
        for idx in layers[-1]:
            neighbors = get_neighbor_idxs(slab, idx, env_strat)
            for neighbor in neighbors:
                if not neighbor in sorted_idxs and not neighbor in new_layer:
                    new_layer.append(neighbor)
        layers.append(new_layer)
        if len(list(sorted_idxs)) >= len(slab.sites):
            break
    return layers

def get_top_layers(slab, n_layers, env_strat=local_env.CrystalNN(), tol=0.5, method='max'):
    '''returns the site indices at the top n layers of a slab'''
    layers = get_layers(slab, env_strat, tol, method)
    idxs = []
    for i in range(n_layers):
        idxs += layers[i]
    return idxs

def gen_input_for_surface_energy_calc(filename, slab, keywords_dict=dict(), num_layers_for_relax=2, 
                                        run_geometry_optimization=True, env_strat=local_env.CrystalNN(), relax_idxs=None):
    '''Computation procedure:
        1. Slab computation:
            - Slab is represented as a unit cell with a \"slab\" in it.
              Unit cell contains large \"vacuum\" areas to seperate different slabs
            - Normal PWDFT for the slab unit cell. The vacuum must be big enough so there won't be
              interaction between slabs.
            - It is recommended to make a relaxation (geometry optimization) at least for the highes
              layer in the slab. To account for the broken bonds.
        2. Normal unit cell computation. No geo-opt.'''
    # setting geo-opt params
    _keywords_dict = deepcopy(keywords_dict)
    if run_geometry_optimization is True:
        if 'ATOMIC_POSITIONS' in keywords_dict.keys():
            # raise warnings.warn("ATOMIC_POSITION specified in keywords_dict, it will be overwritten", Warning, stacklevel=2)
            pass
        _keywords_dict['ATOMIC_POSITIONS'] = np.zeros([len(slab.sites), 3])
        if num_layers_for_relax == 0:
                relax_idxs = []
        if relax_idxs == None:
            relax_idxs = get_top_layers(slab, num_layers_for_relax, env_strat)
        for idx in relax_idxs:
            _keywords_dict['ATOMIC_POSITIONS'][idx] = np.ones(3)
        if not 'calculation' in keywords_dict['CONTROL'].keys():
            _keywords_dict['CONTROL']['calculation'] = 'relax'
        if not 'ion_dynamics' in keywords_dict['IONS'].keys():
            _keywords_dict['IONS']['ion_dynamics'] = 'bfgs'
    WriteInput(slab, filename, _keywords_dict)
    del _keywords_dict
    
def update_slab_from_output_file(filename, slab):
    out_dict = ReadOutput(filename)
    return surface.Slab(Lattice(out_dict['cell_params']), 
                        out_dict['atom_species'], out_dict['atom_pos'],
                        slab.miller_index, struct, slab.shift, slab.scale_factor,)

def slab_from_struct_and_miller(struct, miller_idxs):
    surface.Slab(struct.lattice, struct.species, struct.coords,
                        miller_index, slab.oriented_unit_cell, slab.shift, slab.scale_factor,
                        energy=out_dict['total_energy'])

def calc_energy_of_surface(prefix, working_path, slab, keywords_dict=dict(), qe_path='default', num_layers_for_relax=2, run_geometry_optimization=True, core_num='max', env_strat=local_env.CrystalNN()):
    in_file = os.path.join(working_path, prefix + '.in')
    gen_input_for_surface_energy_calc(in_file, slab, keywords_dict, num_layers_for_relax, run_geometry_optimization, env_strat)
    out_file = os.path.jion(working_path, prefix + '.out')
    if core_num is 'max':
        core_num = os.cpu_count()
    if qe_path is 'default':
        qe_path = os.path.join(os.environ['HOME'], 'q-e-qe-6.5', 'bin')
    RunInput(in_file, out_file, core_num, qe_path)
    return update_slab_from_output_file(out_file, slab)

def FindBestSurface(struct, working_path, keywords_dict=dict(), qe_path='default', slab_size=10, vacuum_size=15, max_idx=3, num_layers_for_relax=2, run_geometry_optimization=True, core_num='max', env_strat=local_env.CrystalNN()):
    '''method to find minimal energy surface of a given pymatgen structure'''
    slabs = generate_slabs(struct, slab_size, vacuum_size, max_idx)
    df = pd.DataFrame(columns=['miller_index', 'total_energy'])
    min_e = 1000
    min_surface = None
    for slab in slabs:
        nslab = calc_energy_of_surface(str(slab.miller_index), working_path, slab, keywords_dict, qe_path, num_layers_for_relax, run_geometry_optimization, core_num, env_strat)
        df = df.append({'miller_index': nslab.miller_index, 'total_energy': nslab.energy})
        if energy < min_e:
            min_surface = nslab
            min_e = nslab.energy
    df.to_csv(os.path.join(working_path, 'results.csv'))
    return min_surface, min_e

def gen_ads_input(path, slab, h, slab_relax_idxs, mol, mol_relax_idxs, binding_position, keywords_dict, run_geometry_optimization=True):
        relax_idxs = copy(slab_relax_idxs)
        num_atoms_in_slab = len(slab.sites)
        vec = np.array(binding_position) + slab.lattice.matrix[-1] / slab.lattice.c * h
        s = slab.copy()
        for site in mol:
            c = site.coords + vec
            s.append(site.specie, c, coords_are_cartesian=True, properties=site.properties)
        if not type(mol_relax_idxs) == list:
            raise ValueError("ilegal mol_relax_idxs value, must be \'all\' or a list of atom indecis in the mol to relax")
        for i in mol_relax_idxs:
            relax_idxs.append(num_atoms_in_slab + i) 
        gen_input_for_surface_energy_calc(path, s, keywords_dict, run_geometry_optimization=run_geometry_optimization, relax_idxs=relax_idxs)
        return s

def GenerateBindInFiles(struct, export_path, miller_idx, slab_layers, vacuum_layers, mol, keywords_dict={}, num_layers_for_relax=2,
                        height=0.5, repeat=None, take_idxs='all',
                        run_geometry_optimization=True, env_strat=local_env.CrystalNN(), prefix='', make_base_ins=True, slab_coords=None, mol_relax_idxs='all'):
    slab = surface.SlabGenerator(struct, miller_idx, slab_layers, vacuum_layers, in_unit_planes=True).get_slabs()[0]
    bind_coords = adsorption.AdsorbateSiteFinder(slab).find_adsorption_sites()['all']
    if not repeat == None:
        slab.make_supercell(repeat)
    if not slab_coords == None:
        d = slab.as_dict()
        d['sites'] = []
        for site, coords in zip(slab.sites, slab_coords):
            s_dict = site.as_dict()
            s_dict['abc'] = coords
            s_dict['xyz'] = slab.lattice.get_cartesian_coords(coords)
            d['sites'].append(s_dict)
        slab = slab.from_dict(d)
        
    # gen base structs for calculation of adsorption energy
    if not os.path.isdir(os.path.join(export_path, 'cif_files')):
        os.mkdir(os.path.join(export_path, 'cif_files'))
    if make_base_ins:
        slab.to("cif", os.path.join(export_path, 'cif_files', prefix + '_base_slab.cif'))
        gen_input_for_surface_energy_calc(os.path.join(export_path, prefix + '_base_slab.in'), slab, keywords_dict, num_layers_for_relax, run_geometry_optimization, env_strat)
        base_mol = mol.get_boxed_structure(slab.lattice.a, slab.lattice.b, slab.lattice.c)
        in_dict = copy(keywords_dict)
        in_dict['ATOMIC_POSITIONS'] = [[1, 1, 1] for i in base_mol.sites]
        WriteInput(base_mol, os.path.join(export_path, prefix + '_base_mol.in'), in_dict)

    relax_idxs = get_top_layers(slab, num_layers_for_relax, env_strat)
    if mol_relax_idxs == 'all':
        mol_relax_idxs = [i for i in range(len(mol.sites))]
    h = height / slab.lattice.c
    for idx, coords in enumerate(bind_coords):
        s = gen_ads_input(os.path.join(export_path, prefix + str(idx) + '.in'), slab, h, relax_idxs, mol, mol_relax_idxs, keywords_dict, run_geometry_optimization)
        s.to("cif", os.path.join(export_path, 'cif_files', prefix + str(idx) + '.cif'))

def gen_slabs_with_defects(slab, repeat=None, num_surface_layers=2, env_strat=local_env.CrystalNN(), tol=0.5):
    remove_coords = [slab.cart_coords[i] for i in get_top_layers(slab, num_surface_layers, env_strat, tol)]
    if not repeat == None:
        slab.make_supercell(repeat)
    structs = []
    for remove_coord in remove_coords:
        s = slab.copy()
        for i, site in enumerate(s.sites):
            if all([round(c1, 4) == round(c2, 4) for c1, c2 in zip(site.coords, remove_coord)]):
                s.remove_sites([i])
        structs.append(s)
    return structs

def binding_inputs_from_out_file(outfile, mol, h, bind_coords, prefix, export_path, num_layers_for_relax=2, mol_relax_idxs='all', keywords_dict={}, run_geometry_optimization=True, coords_are_cartesian=False):
    d = ReadOutput(outfile)
    slab = Structure(d['cell_params'], d['atom_species'], d['atom_pos'])
    if not coords_are_cartesian:
        bind_coords = slab.lattice.get_cartesian_coords(bind_coords)
    if num_layers_for_relax == 0:
      relax_idxs = []
    else:
      relax_idxs = get_top_layers(slab, num_layers_for_relax)
    if mol_relax_idxs == 'all':
        mol_relax_idxs = [i for i in range(len(mol.sites))]
    s = gen_ads_input(os.path.join(export_path, prefix + '.in'), slab, h, relax_idxs, mol, mol_relax_idxs, bind_coords, keywords_dict, run_geometry_optimization)
