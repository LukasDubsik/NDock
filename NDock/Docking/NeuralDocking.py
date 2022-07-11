import torch
from rdkit import Chem
import selfies as sf
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
from Docking.Models import NDocker, Encoder
import sys
import numpy as np
sys.path.append('..')
from Classes import AminoAtom, MoleculeMoves
from Constants import MoleculeEncoder, AminoEncoder


def GetDistance(res, m_x, m_y, m_z):
    return np.sqrt((res.coord_x-m_x)**2+(res.coord_y-m_y)**2+(res.coord_z-m_z)**2)

def GetAminoCenterVector(amino):
    mean_x,mean_y,mean_z = 0,0,0
    for atom in amino.residues:
        mean_x += atom.coord_x
        mean_y += atom.coord_y
        mean_z += atom.coord_z
    return [mean_x/len(amino.residues),mean_y/len(amino.residues),mean_z/len(amino.residues)]
    
def GetAminoDirectionVector(amino):
    m_x,m_y,m_z = GetAminoCenterVector(amino)
    closest = AminoAtom(None, np.inf, np.inf, np.inf, None)
    for residue in amino.residues:
        if GetDistance(residue,m_x,m_y,m_z) < GetDistance(closest,m_x,m_y,m_z):
            closest = residue
    vector = AminoAtom(None,closest.coord_x-m_x, closest.coord_y-m_y, closest.coord_z-m_z, None)
    distance = GetDistance(vector,0,0,0)
    return [vector.coord_x/distance,vector.coord_y/distance,vector.coord_z/distance]

def GetMoleculeMoves(site, molecule):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load('Docking\\NDockEncoder.pth'))
    encoder.eval()
    model = NDocker(encoder).to(device)
    model.load_state_dict(torch.load('Docking\\NDockModel.pth'))
    model.eval()
    
    amino_acids = site.amino
    aminoes = []
    for amino_acid in amino_acids:
        center = GetAminoCenterVector(amino_acid)
        direction_vector = GetAminoDirectionVector(amino_acid)
        holder = 20*[0]
        holder[AminoEncoder[amino_acid.res_name]] = 1
        final = holder+center+direction_vector
        aminoes.append(torch.FloatTensor(final))
        
    mol = molecule.mol
    smiles = Chem.CanonSmiles(Chem.MolToSmiles(mol))
    selfies = sf.encoder(smiles)
    selfies_list = list(sf.split_selfies(selfies))
    point_elements = []
    for element in selfies_list:
        hold_list = [0]*62
        hold_list[MoleculeEncoder[element]] = 1
        point_elements.append(torch.FloatTensor(hold_list))
        
    outputs = model(point_elements, aminoes)
    output = outputs.tolist()[0]
    
    return MoleculeMoves(output[0], output[1], output[2], output[3], output[4], output[5])