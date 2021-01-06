"""
Script to make cif files of all input and output files in a directory. Saves files in a \'results_cifs\' directory at the given path.
USE:
    python results_to_cif.py TARGET_DIR
"""

from utils import *
from pymatgen.core import Structure
import argparse

parser = argparse.ArgumentParser(description='Running parameters')
parser.add_argument('working_path', type=str, help='Path containing all files for the computation. Including cif file for structure.')
args = parser.parse_args()

if not os.path.isdir(os.path.join(args.working_path, 'results_cifs')):
  os.mkdir(os.path.join(args.working_path, 'results_cifs'))

for _file in os.listdir(args.working_path):
  if _file.endswith('.out'):
    print(_file)
    filename = os.path.join(args.working_path, 'results_cifs', _file[:-3] + 'cif')
    d = ReadOutput(os.path.join(args.working_path, _file))
    struct = Structure(d['cell_params'], d['atom_species'], d['atom_pos'])
    struct.to('cif', filename) 