import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdchem
from pathlib import Path


class Site:
    def __init__(self, name, num_lin, num_res, residues):
        self.name = name
        self.num_lin = num_lin
        self.num_res = num_res
        self.residues = residues
    def __str__(self):
        return str(self.name)+", "+str(self.num_lin)+", "+str(self.num_res)
        
class Residue:
    def __init__(self, res_name, chain, res):
        self.res_name = res_name
        self.chain = chain
        self.res = res
    def __str__(self):
        return str(self.res_name)+", "+str(self.chain)+", "+str(self.res)
    
class BaseOperations:
    def GetDistance(self, res, m_x, m_y, m_z):
        return np.sqrt((res.coord_x-m_x)**2+(res.coord_y-m_y)**2+(res.coord_z-m_z)**2)
    
    def FindBaseRotation(self, closest):
        x = closest.coord_x
        y = closest.coord_y
        z = closest.coord_z
        
        if y>0:
            alpha = np.arccos((z)/(np.sqrt(z**2+y**2)))
        elif y<0:
            alpha = 2*np.pi - np.arccos((z)/(np.sqrt(z**2+y**2)))
        else:
            if z>0 or z==0:
                alpha = 0
            elif z<0:
                alpha = np.pi
        
        delta_z = np.sqrt(y**2+z**2)
        if delta_z>0:
            beta = np.arccos((x)/(np.sqrt(z**2+y**2+x**2)))
        elif delta_z<0:
            beta = 2*np.pi - np.arccos((x)/(np.sqrt(z**2+y**2+x**2)))
        else:
            if x>0 or x==0:
                beta = 0
            elif x<0:
                beta = np.pi
    
        return alpha, beta, 0
    
    def Rotate(self,point,alpha,beta,gamma):
        x = point.coord_x
        y = point.coord_y
        z = point.coord_z
    
        alpha_x = x
        alpha_y = y*np.cos(alpha)-z*np.sin(alpha)
        alpha_z = y*np.sin(alpha)+z*np.cos(alpha)
    
        beta_x = alpha_x*np.cos(beta)+alpha_z*np.sin(beta)
        beta_y = alpha_y
        beta_z = -alpha_x*np.sin(beta)+alpha_z*np.cos(beta)
    
        gamma_x = beta_x*np.cos(gamma)-beta_y*np.sin(gamma)
        gamma_y = beta_x*np.sin(gamma)+beta_y*np.cos(gamma)
        gamma_z = beta_z
    
        return AminoAtom(point.local_name, round(gamma_x,4),round(gamma_y,4),round(gamma_z,4), point.name)
    
    def FindCorrector(self,point):
        x = point.coord_x
        y = point.coord_y
        z = point.coord_z
    
        if y>0:
            alpha = np.arccos((z)/(np.sqrt(z**2+y**2)))
        elif y<0:
            alpha = 2*np.pi - np.arccos((z)/(np.sqrt(z**2+y**2)))
        else:
            alpha = np.pi
    
        return alpha, 0, 0
    
    def AddSeparator(str, length):
        sep = ""
        i=0
        while( i < (length-len(str))):
            sep += " "
            i+=1
        return sep
        
class SiteAmino(BaseOperations):
    def __init__(self):
        self.amino = []
    
    def add(self, amino_acid):
        self.amino.append(amino_acid)
        
    def NumAtoms(self):
        counter = 0
        for amino in self.amino:
            counter+=len(amino.residues)
        return counter
    
    def FindDistances(self, m_x, m_y, m_z):
        closest = AminoAtom(None, np.inf, np.inf, np.inf, None)
        farthest = AminoAtom(None, 0, 0, 0, None)
        for amino in self.amino:
            for residue in amino.residues:
                if self.GetDistance(residue,m_x,m_y,m_z) < self.GetDistance(closest,m_x,m_y,m_z):
                    closest = residue
                if self.GetDistance(residue,m_x,m_y,m_z) > self.GetDistance(farthest,m_x,m_y,m_z):
                    farthest = residue
        
        return closest, farthest
    
    def GetRotation(self, transform=False):
        m_x, m_y, m_z = self.GetMean()
        closest, farthest = self.FindDistances(m_x,m_y,m_z)
        
        alpha, beta, gamma = self.FindBaseRotation(closest)
        alpha2, beta2, gamma2 = self.FindCorrector(self.Rotate(farthest,alpha,beta,gamma))
        
        if transform==True:
            self.TransformRotate(alpha, beta, gamma, alpha2, beta2, gamma2)
            
        return alpha, beta, gamma, alpha2, beta2, gamma2
    
    def TransformRotate(self, alpha, beta, gamma, alpha2, beta2, gamma2):
        for i,amino in enumerate(self.amino):
            for j,residue in enumerate(amino.residues):
                residue_new = self.Rotate(self.Rotate(residue, alpha, beta, gamma), alpha2, beta2, gamma2)
                self.amino[i].residues[j] = residue_new
                
    def TransformRotateBase(self, alpha, beta, gamma):
        for i,amino in enumerate(self.amino):
            for j,residue in enumerate(amino.residues):
                residue_new = self.Rotate(residue, alpha, beta, gamma)
                self.amino[i].residues[j] = residue_new
        
    def GetMean(self, transform=False):
        mean_x,mean_y,mean_z = 0,0,0
        count = 0
        
        for amino in self.amino:
            m_x, m_y, m_z = amino.GetMean()
            mean_x += m_x
            mean_y += m_y
            mean_z += m_z
            count += 1
                
        if transform==True:
            self.TransformMean(mean_x/count, mean_y/count, mean_z/count)
            
        return mean_x/count, mean_y/count, mean_z/count
    
    def TransformMean(self, m_x, m_y, m_z):
        for i,amino in enumerate(self.amino):
            for j,residue in enumerate(amino.residues):
                residue_new = AminoAtom(residue.local_name, residue.coord_x - m_x,residue.coord_y- m_y, residue.coord_z - m_z, residue.name)
                self.amino[i].residues[j] = residue_new
                
    def MoveSite(self, transforms):
        self.TransformRotateBase(transforms.alpha, transforms.beta, transforms.gamma)
        self.TransformRotateBase(transforms.alpha2, transforms.beta2, transforms.gamma2)
        self.TransformMean(transforms.m_x, transforms.m_y, transforms.m_z)
        
    def ExportPath(self, export=False):
        if export:
            download_directory = str(Path.home() / "Downloads")+"\\NDockProtein.pdb"
        else:
            download_directory = os.getcwd()+"\\run.pdb"
        f = open(download_directory, "w")
        f = open(download_directory, "a")

        f.write("HEADER    "+"Test1"+"\n")
        f.write("TITLE     "+"Test"+"\n")
        f.write("\n")
        f.write("\n")
        
        count = 1
        counter = 1
        init_amino = self.amino[0]
        for amino in self.amino:
            if amino != init_amino:
                init_amino=amino
                counter+=1
            for res in amino.residues:
                line_init = "ATOM  "+AddSeparator(str(count),5)+str(count)+"  "+res.local_name.upper()+AddSeparator(res.local_name,4)
                line_amino = amino.res_name.upper()+" "+amino.chain.upper()+AddSeparator(str(counter),4)+str(counter)
                line_pos = AddSeparator(str(format(res.coord_x,'.3f')),12)+str(format(res.coord_x,'.3f'))+AddSeparator(str(format(res.coord_y,'.3f')),8)+str(format(res.coord_y,'.3f'))+AddSeparator(str(format(res.coord_z,'.3f')),8)+str(format(res.coord_z,'.3f'))
                line_rest = AddSeparator("1.00",6)+"1.00"+AddSeparator("0.00",6)+"0.00"+"           "+res.name.upper()+AddSeparator(res.name.upper(), 2)+" "
                f.write(line_init+line_amino+line_pos+line_rest+"\n")
                count+=1
        f.write("END")
        f.close()
        return download_directory

class AminoAcid:
    def __init__(self, res_name, chain, res):
        self.residues = []
        self.res_name = res_name
        self.chain = chain
        self.res = res
        
    def add(self, amino_atom):
        self.residues.append(amino_atom)
        
    def GetMean(self):
        mean_x,mean_y,mean_z = 0,0,0
        count = 0
        
        for residue in self.residues:
            mean_x += residue.coord_x
            mean_y += residue.coord_y
            mean_z += residue.coord_z
            count += 1
            
        return mean_x/count, mean_y/count, mean_z/count
        
    def __str__(self):
        return str(self.res_name)+", "+str(self.chain)+", "+str(self.res)
        
class AminoAtom:
    def __init__(self, local_name, x, y, z, name, order=None):
        self.local_name = local_name
        self.coord_x = x
        self.coord_y = y
        self.coord_z = z
        self.name = name
        self.order = order
        
    def __str__(self):
        return str(self.local_name)+", "+str(self.coord_x)+", "+str(self.coord_y)+", "+str(self.coord_z)

class MoleculeBond:
    def __init__(self, atom1, atom2, bond_type):
        self.atom1 = atom1
        self.atom2 = atom2
        self.bond_type = bond_type
    def __str__(self):
        return str(self.atom1)+", "+str(self.atom2)+", "+str(self.bond_type)

class MoleculeAmino(BaseOperations):
    def __init__(self, name, mol):
        self.name = name
        self.atoms = []
        self.bonds = []
        self.mol = mol
        self.closest = None
        self.farthest = None
    
    def add(self, atom):
        self.atoms.append(atom)
        
    def FindDistances(self, m_x, m_y, m_z):
        closest = AminoAtom(None, np.inf, np.inf, np.inf, None)
        farthest = AminoAtom(None, 0, 0, 0, None)
        for i,atom in enumerate(self.atoms):
            if self.GetDistance(atom,m_x,m_y,m_z) < self.GetDistance(closest,m_x,m_y,m_z):
                closest = atom
                closest_num = i
            if self.GetDistance(atom,m_x,m_y,m_z) > self.GetDistance(farthest,m_x,m_y,m_z):
                farthest = atom
                farthest_num = i
        
        return closest, farthest, closest_num, farthest_num
        
    def GetMean(self, transform=False):
        mean_x,mean_y,mean_z = 0,0,0
        count = 0
        
        for atom in self.atoms:
            mean_x += atom.coord_x
            mean_y += atom.coord_y
            mean_z += atom.coord_z
            count += 1
                
        if transform==True:
            self.TransformMean(mean_x/count, mean_y/count, mean_z/count)
            
        return mean_x/count, mean_y/count, mean_z/count
    
    def GetRotation(self, transform=False):
        m_x, m_y, m_z = self.GetMean()
        closest, farthest, i, k = self.FindDistances(m_x,m_y,m_z)
        
        self.closest = i
        self.farthest = k
        
        alpha, beta, gamma = self.FindBaseRotation(closest)
        alpha2, beta2, gamma2 = self.FindCorrector(self.Rotate(farthest,alpha,beta,gamma))
        
        if transform==True:
            self.TransformRotation(alpha, beta, gamma, alpha2, beta2, gamma2)
            
        return alpha, beta, gamma, alpha2, beta2, gamma2
    
    def GetRotationTarget(self, closest, farthest, transform=False):
        m_x, m_y, m_z = self.GetMean()
        
        alpha, beta, gamma = self.FindBaseRotation(closest)
        alpha2, beta2, gamma2 = self.FindCorrector(self.Rotate(farthest,alpha,beta,gamma))
        
        if transform==True:
            self.TransformRotation(alpha, beta, gamma, alpha2, beta2, gamma2)
            
        return alpha, beta, alpha2
        
    def TransformMean(self, m_x, m_y, m_z):
        for i,atom in enumerate(self.atoms):
            atom_new = AminoAtom(atom.local_name, atom.coord_x - m_x,atom.coord_y - m_y, atom.coord_z - m_z, atom.name)
            self.atoms[i] = atom_new
            
    def TransformRotation(self, alpha, beta, gamma, alpha2, beta2, gamma2):
        for i,atom in enumerate(self.atoms):
            self.atoms[i] = self.Rotate(self.Rotate(atom, alpha, beta, gamma), alpha2, beta2, gamma2)
    
    def TransformRotationBase(self, alpha, beta, gamma):
        for i,atom in enumerate(self.atoms):
            self.atoms[i] = self.Rotate(atom, alpha, beta, gamma)
            
    def TransformRotationOther(self, alpha, beta, alpha2):
        for i,atom in enumerate(self.atoms):
            self.atoms[i] = self.Rotate(self.Rotate(atom, alpha, beta, 0), alpha2, 0, 0)
            
    def MoveMolecule(self, moves, transforms):
        self.TransformRotationBase(moves.alpha, moves.beta, moves.gamma)
        self.TransformRotationBase(transforms.alpha, transforms.beta, transforms.gamma)
        self.TransformRotationBase(transforms.alpha2, transforms.beta2, transforms.gamma2)
        self.TransformMean(moves.m_x, moves.m_y, moves.m_z)
        self.TransformMean(transforms.m_x, transforms.m_y, transforms.m_z)
        
    def ExportPath(self, export=False):
        if export:
            download_directory = str(Path.home() / "Downloads")+"\\NDockMolecule.sdf"
        else:
            download_directory = os.getcwd()+"\\run.sdf"
        f = open(download_directory, "w")
        f = open(download_directory, "a")
        
        f.write(self.name+"\n")
        f.write("  -NDock-0.1A"+"\n")
        f.write("\n")
        f.write(AddSeparator(str(len(self.atoms)), 3)+str(len(self.atoms))+AddSeparator(str(len(self.bonds)), 3)+str(len(self.bonds))
            +"  "+"0"+"     "+"0"+"  "+"0"+"  "+"0"+"  "+"0"+"  "+"0"+"  "+"0"+"999 "+"V2000"+"\n")
        
        for atom in self.atoms:
            f.write(AddSeparator(str(format(atom.coord_x,'.4f')), 10)+str(format(atom.coord_x,'.4f'))+AddSeparator(str(format(atom.coord_y,'.4f')), 10)+str(format(atom.coord_y,'.4f'))+AddSeparator(str(format(atom.coord_z,'.4f')), 10)+str(format(atom.coord_z,'.4f'))+AddSeparator(atom.name, 2)+atom.name.upper()+"   "+"0"+"  "+"0"+"  "+"0"+"  "+"0"+"  "+"0"+"  "+ "0"+"  "+"0"+"  "+"0"+"  "+"0"+"  "+"0"+"  "+"0"+"  "+"0"+"\n")
        for bond in self.bonds:
            f.write(AddSeparator(str(bond.atom1), 3)+str(bond.atom1)+AddSeparator(str(bond.atom2), 3)+str(bond.atom2)+AddSeparator(str(bond.bond_type), 3)+str(bond.bond_type)+
                    "  "+"0"+"  "+"0"+"  "+"0"+"  "+"0"+"\n")
            
        f.write("M  END"+"\n")
        f.write("\n")
        f.write("$$$$"+"\n")
    
        f.close()
        return download_directory
            
    def NormalizeEnergy(self, site):
        path_molecule = self.ExportPath()
        path_site = site.ExportPath()
        
        molecule = Chem.MolFromMolFile(path_molecule)
        site_path = open(path_site, 'r').read()
        site = Chem.MolFromPDBBlock(site_path)
        combined = Chem.CombineMols(molecule, site)
        
        combined_hs = Chem.AddHs(combined)
        Chem.SanitizeMol(combined_hs)
        constrained = AllChem.ConstrainedEmbed(combined_hs, combined)
        AllChem.MMFFOptimizeMolecule(constrained)
        constrained = AllChem.ConstrainedEmbed(constrained, combined)
        
        conf = constrained.GetConformers()[0].GetPositions()[0:len(self.atoms)]
        for i in range(0, len(self.atoms)):
            self.atoms[i].coord_x = conf[i][0]
            self.atoms[i].coord_y = conf[i][1]
            self.atoms[i].coord_z = conf[i][2]
        
    def PostTranslate(self, site):
        return None
            
class MoleculeMap:
    def __init__(self):
        self.map = {}
    
    def Map(self, init_atom, goal_atom):
        self.map[init_atom] = goal_atom
    
class Transformation:
    def __init__(self, m_x, m_y, m_z, alpha, beta, gamma, alpha2, beta2, gamma2):
        self.m_x = m_x
        self.m_y = m_y
        self.m_z = m_z
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.alpha2 = alpha2
        self.beta2 = beta2
        self.gamma2 = gamma2
        
class MoleculeMoves:
    def __init__(self, m_x, m_y, m_z, alpha, beta, gamma):
        self.m_x = m_x
        self.m_y = m_y
        self.m_z = m_z
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
def AddSeparator(stri, length):
    sep = ""
    i=0
    while( i < (length-len(stri))):
        sep += " "
        i+=1
    return sep