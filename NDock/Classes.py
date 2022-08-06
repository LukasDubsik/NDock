import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign, rdchem
from pathlib import Path


class Site:
    """
    This class is a part of system of classes that are pass through the intermediates in site extraction process. This class in particular    holds the basic description of the site relevant to pdb file so it could be subseqentiually extracted.
    
    Functions:
        __init__(self, name, num_lin, num_res, residues): Initiates the class. It takes the basic informations of site relevant to pdb file and saves them for further use.
        __str__(self): Outputs the site's information in string formant, used for site analyzis.
            
    Attributes:
        self.name: A string, name of the site, such as AC1.
        self.num_lin: An integer, number of lines by which site is described in SITE header of pdb file.
        self.num_res: An integer, number of residues present in active site.
        self.residues: A list of elements of class Residue indicating position of individual residues in pdb file.
    """
    def __init__(self, name, num_lin, num_res, residues):
        """
        Initiations. It takes as input the properties relevant for the site to be then extracted from the pdb file.
        Parameters:
            Defined in class attributes.
        Returns:
            None
        """
        self.name = name
        self.num_lin = num_lin
        self.num_res = num_res
        self.residues = residues
    def __str__(self):
        """
        Outputs the site's information in string formant, used for site analyzis.
        Paramaters:
            None
        Returns:
            None
        """
        return str(self.name)+", "+str(self.num_lin)+", "+str(self.num_res)
        
class Residue:
    """
    Subclass of the Site class. It holds information about amino acid residue so it can be extracted from the pdb file, this information includes: amino acid residue (ILE, ALA), chain name (A, B, C) and position on that chain (1,2,3).
    
    Functions:
        __init__(self, res_name, chain, res): Initiates the class. It adds the given values to class atrributes for further accesibility.
        __str__(self): Outputs the residue's information in string formant, used for analyzis.
            
    Attributes:
        self.res_name: A string, amino acid residue (ILE, ALA).
        self.chain: A string, the letter indicating protein domain (A, B, C).
        self.res: A string, position of residue relevant to given protein domain.
    """
    def __init__(self, res_name, chain, res):
        """
        Initiations. It takes as input the properties relevant for the residue to be then extracted from the pdb file.
        Parameters:
            Defined in class attributes.
        Returns:
            None
        """
        self.res_name = res_name
        self.chain = chain
        self.res = res
    def __str__(self):
        """
        Outputs the residue's information in string formant, used for analyzis.
        Paramaters:
            None
        Returns:
            None
        """
        return str(self.res_name)+", "+str(self.chain)+", "+str(self.res)
    
class BaseOperations:
    """
    Holds important functions used to manipulate both the active site and the ligand and as such both the MoleculeAmino and SiteAmino inherit from this class.
    Functions:
        GetDistance(self, res, m_x, m_y, m_z): Gets distance between atom and mean.
        FindBaseRotation(self, closest): Finds rotation around atom so that this atom has coordinates of (0,y,0). This atom is generally the closest atom to the molecule center, but can be interchanged. This functions assumes that molecule has already been normalized with center at origin.
        FindCorrector(self,point): Finds additinal rotational correction, that means, normalizes molecule around y-axis. The point is generally molecule farthest from the center. Again assumes the center normalized to origin.
        Rotate(self,point,alpha,beta,gamma): Rotates the molecule by the angles specified. These are generally extracted from the FindBaseRotation and FindCorrector.
            
    Attributes:
        None
    """
    def GetDistance(self, res, m_x, m_y, m_z):
        """
        Function to get distance between atom and mean of atom's molecule.
        
        Parameters:
            res: Class AminoAtom, atom to get distance from.
            m_x, m_y, m_y: Floats, centers of the molecule.
        
        Returns:
            distance: Gives disatnce between both res and m_x-z.
        """
        distance = np.sqrt((res.coord_x-m_x)**2+(res.coord_y-m_y)**2+(res.coord_z-m_z)**2)
        return distance
    
    def FindBaseRotation(self, closest):
        """
        Finds rotation around atom so that this atom has coordinates of (0,y,0) (Used for both ligand and active site normalization). This atom is generally the closest atom to the molecule center, but can be interchanged. This functions assumes that molecule has already been normalized with center at origin.
        
        Parameters:
            closest: Class AminoAtom, specifies the atom which is to be rotated to the (0,y,0) coordinates.
        
        Returns:
            alpha, beta, 0: Floats, angles neccesary to rotate all atoms to achive the given atom desired coordinates.
        """
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
    
        return alpha, beta, 0.0
    
    def FindCorrector(self, point):
        """
        Finds additinal rotational correction, that means, normalizes molecule around y-axis. This normalization's goal is to get the atom's coordinates to (x,0,z) The point is generally molecule farthest from the center. Again assumes the center normalized to origin. This, coupled with above FindBaseRotation should remove all rotational variance making all molecules, no matter their origin, have same coordinates.
        
        Parameters:
            point: lass AminoAtom, specifies the atom which is to be rotated to the (x,0,z) coordinates.
        
        Returns:
            alpha,0,0: Floats, angles neccesary to rotate all atoms to achive the given atom desired coordinates.
        """
        x = point.coord_x
        y = point.coord_y
        z = point.coord_z
    
        if y>0:
            alpha = np.arccos((z)/(np.sqrt(z**2+y**2)))
        elif y<0:
            alpha = 2*np.pi - np.arccos((z)/(np.sqrt(z**2+y**2)))
        else:
            alpha = np.pi
    
        return alpha, 0.0, 0.0
    
    def Rotate(self,point,alpha,beta,gamma):
        """
        Rotates the molecule by the angles specified. These are generally extracted from the FindBaseRotation and FindCorrector.
        
        Parameters:
            point: Class AminoAtom, point to be rotated by given angles.
            alpha,beta,gamma: Floats, angles that are used to rotate the given atom.
        
        Returns:
            rotated_atom: Class AminoAtom, atom that has been rotated.
        """
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
        
        rotated_atom = AminoAtom(point.local_name, round(gamma_x,4),round(gamma_y,4),round(gamma_z,4), point.name)
    
        return rotated_atom
        
class SiteAmino(BaseOperations):
    """
    This class is the main class describing the active site extracted from the protein. it holds the elements of the active site, its amino acids, with further functions used to modify the site or work with it in another context (such as export to pdb file).
    
    Functions:
        __init__(self): Initiates the class. Its only function is to create the list of amino acids.
        add(self, amino_acid): Adds the desired amino atom of class AminoAcid to the self.amino.
        NumAtoms(self): Helper function, gives the number of atoms per all amino acids.
        FindDistances(self, m_x, m_y, m_z): Used to find the closest and farthest atom to the center of the active site. These are then subsequently used for rotational normalization.
        GetRotation(self, transform=False): Returns both the base and correctional rotationas needed for rotational normalization. If transform set to True also transforms the class.
        TransformRotate(self, alpha, beta, gamma, alpha2, beta2, gamma2): Rotates every atom in molecule by given angles.
        TransformRotateBase(self, alpha, beta, gamma): Rotates every atom in molecule by given angles.
        GetMean(self, transform=False): Gets mean of the active site, if transform set to True, also normalizes the class by the mean.
        TransformMean(self, m_x, m_y, m_z): Normalizes the molecule's mean to the origin.
        MoveSite(self, transforms): Uses the transforms (both the rotational and mean moves) to move the site by specified constants.
        ExportPath(self, export=False): Exports the pdb file of the active site.
            
    Attributes:
        self.amino: A list, contains the amino acids of the site of the class AminoAcid. 
    """
    def __init__(self):
        """
        Initiates the class. Its only function is to create the list of amino acids.
        Parameters:
            Defined in class attributes.
        Returns:
            None
        """
        self.amino = []
    
    def add(self, amino_acid):
        """
        Adds the desired amino atom of class AminoAcid to the self.amino.
        
        Parameters:
            amino_acid: Class AminoAcid, amino acid to be added to the list of amino acids constituting the site.
        
        Returns:
            Nothing, only adds the amino acid.
        """
        self.amino.append(amino_acid)
        
    def NumAtoms(self):
        """
        Helper function, gives the number of atoms per all amino acids.
        
        Parameters:
            None
        
        Returns:
            counter: Int, number of atoms in active site.
        """
        counter = 0
        for amino in self.amino:
            counter+=len(amino.residues)
        return counter
    
    def FindDistances(self, m_x, m_y, m_z):
        """
        Used to find the closest and farthest atom to the center of the active site (or other point in space if wished). These are then subsequently used for rotational normalization. Utilizes the GetDistance function inherited from BaseOperation class.
        
        Parameters:
            m_x,m_y,m_z: Floats, coordinates of the center of the amino acid (or other point).
        
        Returns:
            closest: Class AminoAtom, the atom from active site closest the center.
            farthest: Class AminoAtom, the atom from active site farthest from the center.
        """
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
        """
        Returns both the base and correctional rotationas needed for rotational normalization. If transform set to True also transforms the class. This is done by using the closest atom to the molecule's center (mean) and then rotating it so that its coordinates are (0,y,0) and the farthest has coordinates (x,0,z). These rotations serve to eliminate all rotational variance.
        
        Parameters:
            transform=False: Boolean, indicates, if resulting rotations should be applied to rotating molecule.
        
        Returns:
            alpha, beta, gamma, alpha2, beta2, gamma2: Floats, both base and corrector angles used for site normalization.
        """
        m_x, m_y, m_z = self.GetMean()
        closest, farthest = self.FindDistances(m_x,m_y,m_z)
        
        alpha, beta, gamma = self.FindBaseRotation(closest)
        alpha2, beta2, gamma2 = self.FindCorrector(self.Rotate(farthest,alpha,beta,gamma))
        
        if transform==True:
            self.TransformRotate(alpha, beta, gamma, alpha2, beta2, gamma2)
            
        return alpha, beta, gamma, alpha2, beta2, gamma2
    
    def TransformRotate(self, alpha, beta, gamma, alpha2, beta2, gamma2):
        """
        Rotates every atom in molecule by given angles. This function is variation to TransformRotateBase, which uses only three angles.
        
        Parameters:
            alpha, beta, gamma, alpha2, beta2, gamma2: Floats, both base and corrector angles used for site normalization.
        
        Returns:
            Nothing, just changes coordinates of every atom in active site.
        """
        for i,amino in enumerate(self.amino):
            for j,residue in enumerate(amino.residues):
                residue_new = self.Rotate(self.Rotate(residue, alpha, beta, gamma), alpha2, beta2, gamma2)
                self.amino[i].residues[j] = residue_new
                
    def TransformRotateBase(self, alpha, beta, gamma):
        """
        Rotates every atom in molecule by given angles. This function is variation to TransformRotate, which uses six angles.
        
        Parameters:
            alpha, beta, gamma: Floats, both base and corrector angles used for site normalization.
        
        Returns:
            Nothing, just changes coordinates of every atom in active site.
        """
        for i,amino in enumerate(self.amino):
            for j,residue in enumerate(amino.residues):
                residue_new = self.Rotate(residue, alpha, beta, gamma)
                self.amino[i].residues[j] = residue_new
        
    def GetMean(self, transform=False):
        """
        Gets mean of the active site, if transform set to True, also normalizes the class by the mean. This mean is just an average of all amino acids present.
        
        Parameters:
            transform=False: Boolean, indicates, if resulting mean should be substracted from the active site center, moving it to origin.
        
        Returns:
            mean_x, mean_y, mean_z: Center of the active site (if transform was applied, then values for reconstructing previous center).
        """
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
        
        mean_x = mean_x/count
        mean_y = mean_y/count
        mean_z = mean_z/count
        
        return mean_x, mean_y, mean_z
    
    def TransformMean(self, m_x, m_y, m_z):
        """
        Normalizes the molecule's mean to the origin. If different values than mean are used, moves active site's center.
        
        Parameters:
            m_x,m_y,m_z: Floats, coordinates of the center of the amino acid (or other point).
        
        Returns:
            Nothing, transforms coordinates of every atom in active site.
        """
        for i,amino in enumerate(self.amino):
            for j,residue in enumerate(amino.residues):
                residue_new = AminoAtom(residue.local_name, residue.coord_x - m_x,residue.coord_y- m_y, residue.coord_z - m_z, residue.name)
                self.amino[i].residues[j] = residue_new
                
    def MoveSite(self, transforms):
        """
        Uses the transforms (both the rotational and mean moves) to move the site by specified constants.
        
        Parameters:
            transforms: Class Transformations, includes both the base, correctinal and center normalization.
        
        Returns:
            Nothing, moves active site to its original location for final export by using the constants used in its normalization.
        """
        self.TransformRotateBase(transforms.alpha, transforms.beta, transforms.gamma)
        self.TransformRotateBase(transforms.alpha2, transforms.beta2, transforms.gamma2)
        self.TransformMean(transforms.m_x, transforms.m_y, transforms.m_z)
        
    def ExportPath(self, export=False):
        """
        Exports the pdb file of the active site. This includes both the HEADER, TITLE and list of all atoms in order present.
        
        Parameters:
            export=False: Boolean, if set to True the file will be exported to the Download folder, otherwise it will end up in the program directory.
        
        Returns:
            download_directory: String, path to the file's directory (program's directory or download, depends on the export parameter).
        """
        if export:
            download_directory = str(Path.home() / "Downloads")+"\\NDockProtein.pdb"
        else:
            download_directory = os.getcwd()+"\\run.pdb"
        f = open(download_directory, "w")
        f = open(download_directory, "a")

        f.write("HEADER    "+"ACTIVESITE"+"\n")
        f.write("TITLE     "+"ACTIVE SITE STRUCTURE BY PROGRAM NDOCK"+"\n")
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
    """
    Class representing amino acid in protein. Includes values necessary to reconstruct amino acid and also some functions to get values.
    
    Functions:
        __init__(self, res_name, chain, res): Initiates the class. Takes the basic informations from the amino acid pdb and ads them to class variables
        add(self, amino_atom): Adds atom that is extracted from the pdb file to construct amino acid.
        GetMean(self): Returns the mean of atoms contained in the amino acid.
        __str_(str): Returns string representation of the class for better analysis.
            
    Attributes:
        self.residues: A list, contains list of atoms from pdb file.
        self.res_name: A string, name of the residue, such as GLU or ALA.
        self.chain: A string, chain of protein, such as A or B,C.
        self.res: An int, position upon the chain from its C end.
    """
    def __init__(self, res_name, chain, res):
        """
        Initiations. It takes as input the properties relevant for the amino acid to be then extracted from the pdb file.
        Parameters:
            Defined in class attributes.
        Returns:
            Nothing, just assignings variables.
        """
        self.residues = []
        self.res_name = res_name
        self.chain = chain
        self.res = res
        
    def add(self, amino_atom):
        """
        Function to add atoms constructing the amino acid in their extracted form of AminoAtom class.
        
        Parameters:
            amino_atom: An AminoAtom class, class containing information about atoms, both from pdb and sdf files.
        
        Returns:
            Nothing, just adds atoms to class's variable self.residues.
        """
        self.residues.append(amino_atom)
        
    def GetMean(self):
        """
        Returns mean of atoms present in the list.
        Parameters:
            None, class variables are used.
        Returns:
            Means in x, y, z directions, all are floats.
        """
        mean_x,mean_y,mean_z = 0,0,0
        count = 0
        
        for residue in self.residues:
            mean_x += residue.coord_x
            mean_y += residue.coord_y
            mean_z += residue.coord_z
            count += 1
        
        return mean_x/count, mean_y/count, mean_z/count
        
    def __str__(self):
        """
        Outputs the residue's information in string formant, used for analysis.
        
        Paramaters:
            None, class variables are used.
        
        Returns:
            Class's representation in string.
        """
        return str(self.res_name)+", "+str(self.chain)+", "+str(self.res)
        
class AminoAtom:
    """
    Class to hold atom representation, used for both pdb and sdf file extraction. Constains atom's basic informations for its reconstruction and also method for string representation. 
    
    Functions:
        __init__(self, local_name, x, y, z, name, order=None): Initiates the class. It needs local name (for pdf file), coordinates (x,y,z), its atom name (such as C, N, O...) and if desired, order (sdf files).
        __str__(self): Outputs the atom's information in string formant, used for analysis.
            
    Attributes:
        self.local_name: A string, local atom's name for pdb files.
        self.coord_x: A float, atom's coordinate in x dimension.
        self.coord_y: A float, atom's coordinate in y dimension.
        self.coord_z: A float, atom's coordinate in z dimension.
        self.name: A string, atom's chemical name, such as C, N or O.
        self.order: An int, if desired, used for sdf files.
    """
    def __init__(self, local_name, x, y, z, name, order=None):
        """
        Initiations. It takes as input the properties relevant for the atom to be then extracted from the pdbor sdf file.
        Parameters:
            Defined in class attributes.
        Returns:
            Nothing, just initializes the atom's parameters.
        """
        self.local_name = local_name
        self.coord_x = x
        self.coord_y = y
        self.coord_z = z
        self.name = name
        self.order = order
        
    def __str__(self):
        """
        Outputs the residue's information in string formant, used for analysis.
        
        Paramaters:
            None, class variables are used.
        
        Returns:
            Class's representation in string.
        """
        return str(self.local_name)+", "+str(self.coord_x)+", "+str(self.coord_y)+", "+str(self.coord_z)

class MoleculeBond:
    """
    Class to hold necessary information for molecule's bond extracted from sdf file. These informations are used to reconstruct this bond in reference to MoleculeAmino class, which contains informations relating to the atoms themselves.
    
    Functions:
        __init__(self, atom1, atom2, bond_type): Initiates the class. As input it takes position of atoms in reference to sdf files, more precise information are then given in related MoleculeAmino class and type of bond.
        __str__(self): Returns string representation for further analysis. 
            
    Attributes:
        self.atom1: An int, position of atom relative to sdf file, more informations can be extracted from related class MoleculeAmino.
        self.atom2: An int, position of atom relative to sdf file, more informations can be extracted from related class MoleculeAmino.
        self.bond_type: A string, type of bond's type -> 1: single, 2: double, 3: triple, 4->aromatic.
    """
    def __init__(self, atom1, atom2, bond_type):
        """
        Initiations. It takes as input the properties relevant for the bond to be then extracted from the sdf file.
        Parameters:
            Defined in class attributes.
        Returns:
            Nothing, just initializes the class.
        """
        self.atom1 = atom1
        self.atom2 = atom2
        self.bond_type = bond_type
    def __str__(self):
        """
        Outputs the bond's information in string formant, used for analysis.
        
        Paramaters:
            None, class variables are used.
        
        Returns:
            Class's representation in string.
        """
        return str(self.atom1)+", "+str(self.atom2)+", "+str(self.bond_type)

class MoleculeAmino(BaseOperations):
    """
    Class containing information to reconstruct molecule that has been extracted from the sdf file. Furthemore, it also include all the necessary functions to preprocess and then use this molecule in NDock and complementary functions.
    
    Functions:
        def __init__(self, name, mol):Initiates the class. Its function is to create the class variables which are described below.
        def add(self, atom): Adds the class AminoAtoms atoms to create the molecule.
        def FindDistances(self, m_x, m_y, m_z): Used to find the closest and farthest atom to the center of the molecule. These are then subsequently used for rotational normalization.
        def GetMean(self, transform=False): Gets mean of the molecule, if transform set to True, also normalizes the class by the mean.
        def GetRotation(self, transform=False): Returns both the base and correctional rotationas needed for rotational normalization. If transform set to True also transforms the class.
        def GetRotationTarget(self, closest, farthest, transform=False): Returns both the base and correctional rotations needed for rotational normalization of target. If transform set to True also transforms the class.
        def TransformMean(self, m_x, m_y, m_z): Normalizes the molecule's mean to the origin.
        def TransformRotation(self, alpha, beta, gamma, alpha2, beta2, gamma2): Rotates every atom in molecule by given angles.
        def TransformRotationBase(self, alpha, beta, gamma): Rotates every atom in molecule by given angles.
        def TransformRotationOther(self, alpha, beta, alpha2): Rotates every atom in molecule by given angles for target.
        def MoveMolecule(self, moves, transforms): Uses the transforms (both the rotational and mean moves) to move the site by specified constants.
        def ExportPath(self, export=False): Exports the sdf file of the active site.
        def NormalizeEnergy(self, site): Normalizes energy of the molecule.
        def PostTranslate(self, site): Currently disfunctional.
            
    Attributes:
        self.name: A string, name of the molecule.
        self.atoms: A list, contains atoms of class AminoAtom.
        self.bonds: A list, constainsatoms of class MoleculeBond.
        self.mol: A string, path to sdf file.
        self.closest: A AminoAtom, position of closest atom relative.
        self.farthest: A AminoAtom, position of farthest atom relative.
    """
    def __init__(self, name, mol):
        """
        Initiations. It creates the class variables for further reference in class functions.
        Parameters:
            Defined in class attributes.
        Returns:
            Nothing, just initializes the parameters.
        """
        self.name = name
        self.atoms = []
        self.bonds = []
        self.mol = mol
        self.closest = None
        self.farthest = None
    
    def add(self, atom):
        """
        Adds the desired atom of class AminoAtom to the self.atoms.
        
        Parameters:
            atom: Class AminoAtom, atom to be added to the list of atoms.
        
        Returns:
            Nothing, only adds the atom.
        """
        self.atoms.append(atom)
        
    def FindDistances(self, m_x, m_y, m_z):
        """
        Used to find the closest and farthest atom to the center of the molecule (or other point in space if wished). These are then subsequently used for rotational normalization. Utilizes the GetDistance function inherited from BaseOperation class.
        
        Parameters:
            m_x,m_y,m_z: Floats, coordinates of the center of the molecule (or other point).
        
        Returns:
            closest: Class AminoAtom, the atom from active site closest the center.
            farthest: Class AminoAtom, the atom from active site farthest from the center.
        """
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
        """
        Gets mean of the molecule, if transform set to True, also normalizes the class by the mean. This mean is just an average of all atoms present.
        
        Parameters:
            transform=False: Boolean, indicates, if resulting mean should be substracted from the molecule's center, moving it to origin.
        
        Returns:
            mean_x, mean_y, mean_z: Center of the molecule (if transform was applied, then values for reconstructing previous center).
        """
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
        """
        Returns both the base and correctional rotationas needed for rotational normalization. If transform set to True also transforms the class. This is done by using the closest atom to the molecule's center (mean) and then rotating it so that its coordinates are (0,y,0) and the farthest has coordinates (x,0,z). These rotations serve to eliminate all rotational variance.
        
        Parameters:
            transform=False: Boolean, indicates, if resulting rotations should be applied to rotating molecule.
        
        Returns:
            alpha, beta, gamma, alpha2, beta2, gamma2: Floats, both base and corrector angles used for site normalization.
        """
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
        """
        Returns both the base and correctional rotationas needed for rotational normalization. If transform set to True also transforms the class. This is done by using the closest atom to the molecule's center (mean) and then rotating it so that its coordinates are (0,y,0) and the farthest has coordinates (x,0,z). These rotations serve to eliminate all rotational variance. Used for target.
        
        Parameters:
            transform=False: Boolean, indicates, if resulting rotations should be applied to rotating molecule.
        
        Returns:
            alpha, beta, gamma, alpha2, beta2, gamma2: Floats, both base and corrector angles used for site normalization.
        """
        m_x, m_y, m_z = self.GetMean()
        
        alpha, beta, gamma = self.FindBaseRotation(closest)
        alpha2, beta2, gamma2 = self.FindCorrector(self.Rotate(farthest,alpha,beta,gamma))
        
        if transform==True:
            self.TransformRotation(alpha, beta, gamma, alpha2, beta2, gamma2)
            
        return alpha, beta, alpha2
        
    def TransformMean(self, m_x, m_y, m_z):
        """
        Normalizes the molecule's mean to the origin. If different values than mean are used, moves active site's center.
        
        Parameters:
            m_x,m_y,m_z: Floats, coordinates of the center of the atom (or other point).
        
        Returns:
            Nothing, transforms coordinates of every atom in molecule.
        """
        for i,atom in enumerate(self.atoms):
            atom_new = AminoAtom(atom.local_name, atom.coord_x - m_x,atom.coord_y - m_y, atom.coord_z - m_z, atom.name)
            self.atoms[i] = atom_new
            
    def TransformRotation(self, alpha, beta, gamma, alpha2, beta2, gamma2):
        """
        Rotates every atom in molecule by given angles. This function is variation to TransformRotateBase, which uses only three angles.
        
        Parameters:
            alpha, beta, gamma, alpha2, beta2, gamma2: Floats, both base and corrector angles used for parameter normalization.
        
        Returns:
            Nothing, just changes coordinates of every atom in molecule.
        """
        for i,atom in enumerate(self.atoms):
            self.atoms[i] = self.Rotate(self.Rotate(atom, alpha, beta, gamma), alpha2, beta2, gamma2)
    
    def TransformRotationBase(self, alpha, beta, gamma):
        """
        Rotates every atom in molecule by given angles. This function is variation to TransformRotate, which uses six angles.
        
        Parameters:
            alpha, beta, gamma: Floats, both base and corrector angles used for molecule normalization.
        
        Returns:
            Nothing, just changes coordinates of every atom in molecule.
        """
        for i,atom in enumerate(self.atoms):
            self.atoms[i] = self.Rotate(atom, alpha, beta, gamma)
            
    def TransformRotationOther(self, alpha, beta, alpha2):
        """
        Rotates every atom in molecule by given angles. This function is variation to TransformRotate, which uses six angles. Used for target.
        
        Parameters:
            alpha, beta, gamma: Floats, both base and corrector angles used for site normalization.
        
        Returns:
            Nothing, just changes coordinates of every atom in molecule.
        """
        for i,atom in enumerate(self.atoms):
            self.atoms[i] = self.Rotate(self.Rotate(atom, alpha, beta, 0), alpha2, 0, 0)
            
    def MoveMolecule(self, moves, transforms):
        """
        Uses the transforms (both the rotational and mean moves) to move the molecule by specified constants.
        
        Parameters:
            transforms: Class Transformations, includes both the base, correctinal and center normalization.
        
        Returns:
            Nothing, moves molecule to its original location for final export by using the constants used in its normalization.
        """
        self.TransformRotationBase(moves.alpha, moves.beta, moves.gamma)
        self.TransformRotationBase(transforms.alpha, transforms.beta, transforms.gamma)
        self.TransformRotationBase(transforms.alpha2, transforms.beta2, transforms.gamma2)
        self.TransformMean(moves.m_x, moves.m_y, moves.m_z)
        self.TransformMean(transforms.m_x, transforms.m_y, transforms.m_z)
        
    def ExportPath(self, export=False):
        """
        Exports the sdf file of the molecule. This includes all the necessary informations for visualization and further work.
        
        Parameters:
            export=False: Boolean, if set to True the file will be exported to the Download folder, otherwise it will end up in the program directory.
        
        Returns:
            download_directory: String, path to the file's directory (program's directory or download, depends on the export parameter).
        """
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
        """
        Changes the realtiveposition of molecule and active site so as to lower the free energy. This function currently uses the rdkit functions which are not so precise in molecule aligmentso caution is required when using this function.
        
        Parameters:
            site: SiteAmino class, contains list of all amino acids and their properties present in the protein's site.
            
        Returns:
            Nothing, just changes the position of molecule relative to active site.
        """
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
        """
        Currently not accesible.
        """
        return None
            
class MoleculeMap:
    """
    Serves to map atoms of different molecules when trying to align them (this is due to different types of writing atoms order in sdf files and pdb files). Used in reference to the MoleculeAmino class.
    
    Functions:
        __init__(self): Initiates the class. Does not take parameters, just holds dictionary for further accesibility.
        Map(self, init_atom, goal_atom): Adds entry to the dictionary, key is init_atom.
            
    Attributes:
        self.map: A dictionary containing the map of atoms between two different molecules.
    """
    def __init__(self):
        """
        Initiations. It takes as input the properties relevant for the site to be then extracted from the pdb file.
        Parameters:
            Defined in class attributes.
        Returns:
            None
        """
        self.map = {}
    
    def Map(self, init_atom, goal_atom):
        """
        Adds entries from the init molecule, the init atoms, as key to its value from goal molecule, the goal atoms.
        
        Parameters:
            init_atom: A float, refers to position of atom in a list of a MoleculeAtom class of the init Molecule.
            goal_atom: A float, refers to position of atom in a list of a MoleculeAtom class of the goal Molecule.
        
        Returns:
            Nothing, just modifies the self.map dictionary.
        """
        self.map[init_atom] = goal_atom
    
class Transformation:
    """
    Holds the moves given when normalizing the molecule or active site of protein. The angles without number subscript are for first rotational normalization, while the ones with subscript for the second, further description in class BaseOperations. It is used for ligand and protein reconstruction after docking to original positions.
    
    Functions:
        __init__(self, m_x, m_y, m_z, alpha, beta, gamma, alpha2, beta2, gamma2): Initiates the class. Takes the parameters given by normalization.
            
    Attributes:
        self.m_x: A float, indicates move in x direction in space.
        self.m_y: A float, indicates move in y direction in space.
        self.m_z: A float, indicates move in z direction in space.
        self.alpha: A float, rotation around x-axis in radians for first normalization.
        self.beta: A float, rotation around y-axis in radians for first normalization.
        self.gamma: A float, rotation around z-axis in radians for first normalization.
        self.alpha2: A float, rotation around x-axis in radians for second normalization.
        self.beta2: A float, rotation around y-axis in radians for second normalization.
        self.gamma2: A float, rotation around z-axis in radians for second normalization.
    """
    def __init__(self, m_x, m_y, m_z, alpha, beta, gamma, alpha2, beta2, gamma2):
        """
        Initiations. It takes the parameters given by normalization.
        Parameters:
            Defined in class attributes.
        Returns:
            None
        """
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
    """
    Holds the moves of the molecule as outputed from the neural site.
    
    Functions:
        __init__(self, m_x, m_y, m_z, alpha, beta, gamma): Initiates the class. It takes the moves and rotational angles as given by Neural site.
            
    Attributes:
        self.m_x: A float, indicates move in x direction in space.
        self.m_y: A float, indicates move in y direction in space.
        self.m_z: A float, indicates move in z direction in space.
        self.alpha: A float, rotation around x-axis in radians.
        self.beta: A float, rotation around y-axis in radians.
        self.gamma: A float, rotation around z-axis in radians.
    """
    def __init__(self, m_x, m_y, m_z, alpha, beta, gamma):
        """
        Initiations. It takes moves given from neural site of NDock.
        Parameters:
            Defined in class attributes.
        Returns:
            None
        """
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