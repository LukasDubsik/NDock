import sys
import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdchem
sys.path.append('..')
from Classes import MoleculeAmino, MoleculeBond, AminoAtom

def NormalizeMolecule(molecular_path):
    
    transformation_coord = molecular_path.split("\\")[-1].split("_")[0][-2]
    
    file = open(molecular_path, 'r').read()
    mol_file = Chem.MolFromMolBlock(file, removeHs=True)
    if transformation_coord=="2":
        mol_fil = Chem.AddHs(mol_file)
        AllChem.EmbedMolecule(mol_fil)
        AllChem.UFFOptimizeMolecule(mol_fil)
        mol_fi = Chem.MolToMolBlock(mol_fil)
        mol_file = Chem.MolFromMolBlock(mol_fi, removeHs=True)
    mal_3D = Chem.MolToMolBlock(mol_file)
       
    molecular_path = os.getcwd()+"\\run_file.sdf"
    m = open(molecular_path, "w")
    m = open(molecular_path, "a")
    m.write(mal_3D)
    m.close()
    
    with open(molecular_path) as file:
        molecule_lines = file.readlines()
        
    info_line = molecule_lines[3]
    num_atm = int(info_line[0:3])
    num_bonds = int(info_line[3:6])
    molecule = MoleculeAmino(molecule_lines[0].strip().lower(), Chem.MolFromMolFile(molecular_path, removeHs = True))
    
    for i,line in enumerate(molecule_lines[4:num_atm+4]):
        name_atom = line[31:33].strip().lower()
        atom = AminoAtom(None, float(line[0:10]), float(line[10:20]), float(line[20:30]), name_atom, i)
        molecule.add(atom)
        
    for line in molecule_lines[4+num_atm:4+num_atm+num_bonds]:
        molecule.bonds.append(MoleculeBond(int(line[0:3]),int(line[3:6]),int(line[6:9])))
        
    molecule.GetMean(transform=True)
    if num_atm>1:
        molecule.GetRotation(transform=True)
    
    os.remove(molecular_path)
    
    return molecule