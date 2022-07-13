from Preprocessing import NormalizeMolecule
from Preprocessing import NormalizeSite
from Docking import NeuralDocking
from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem, rdMolAlign, rdchem, rdEHTTools
from NDock.Constants import Well_depth, Walls_Radiuses, constants, Atomic_radiuses, Solvation


class NDock:
    """
    NDock class serves to dock ligands (small molecules reacting with proteins) into specified 
    proteins. It relies on neural sites as its mean of achieving this. In use are also energy 
    normalization and strain removal. 
    
    Functions:
        __init__(self, path_to_pdb): Initiates the class. It takes path to protein file (currently only pdb, 
            soon mmcif), from which it then excrates lines with whom further functions work.
        Dock(self, path_to_sdf, site_name, site_simplification=False, energy_normalization=True, post_translation=True): 
            Takes as input path to sdf (molecule to be docked) file and site name of previously specified protein.
            This function performs the docking itself. It firstly normalizes input, then uses neural site and, if 
            specified, another functionalities.
            
    Attributes:
        self.path_to_pdb: A string indicating path to specified pdb file.
        self.pdb_lines: A list containing the extracted lines of the pdb file.
    """
    def __init__(self, path_to_pdb):
        """
        This function initiates the NDock class by taking path to protein (currently only pdb) as its input.
        It then uses this path to extract pdb lines which are further used the Dock function.
        
        Parameters:
            path_to_pdb: String which specified a path to pdb file of protein for docking.
        
        Returns:
            None
        """
        self.path_to_pdb = path_to_pdb
        
        with open(path_to_pdb) as pdb_file:
            self.pdb_lines = pdb_file.readlines()
            
    def Dock(self, path_to_sdf, site_name, site_simplification=False, energy_normalization=False, post_translation=False):
        """
        Purpose of Dock function is to perform the molecular docking itself.
        
        After protein, its active site and molecule, which is to be docked, are specified by the user NDock
        firstly normalizes both the site and the molecule to mean zero and removes all rotational variation 
        (this is used to make all molecules of same composition have same starting point). Molecule is
        converted to selfies and each amino acid of active site to its amino type (one hot encoding) and its
        rotational and positional vectors extracted. These then go through layers of linear neural networks, 
        outputting 6 numbers (move in three directions and rotations around 3 angles in three dimensional 
        space). Result can then be energicaly normalised, so that its free energy is at its lowest, or 
        its strain can be minimalised.
        
        Parameters:
            path_to_sdf: A string indicating path to a sdf file contatininf the molecule to be docked.
            site_name: Name of site present in the protein file, such as AC1, ac3 ec. Alternative is a 
                list of amino acids present in the site, this is preferable as it removes the risk of 
                wrongly assigned site. Example: [[Arg, a, 250], [Phe, a, 76]] ec.
            site_simplification: Boolean, if set to True, the function would automatticaly remove all amino acids 
                too far from the main concetration of other amino acids, making the site more compactfull.
                This only serves as last ditch effort to make the result more precise and it is strongly
                advised for user to prepare the site before entering here.
            energy_normalization: Boolean, if set to True molecules free energy will be normalized. This could lead
                to better results, but as rdkit's Constrained embedding and optimization are not perfect, it could result
                in sizezable error an also requires substantial time. It is recommended to compare the results 
                before deciding if to use. In future, own functions will be used with better result expected.
            post_translation: Boolean. Currently unaviable, as there is a problem in orbital overlaping 
                computations.
                
        Returns:
            path_to_result: String, returns path to sdf molecule result with its atomic coordinates mirroring
                the ones in given site upon docking.
            results: Class NDockResult, this class holds values extracted from the docking and some functions,
                which use these results, such as score or electrical charges computing.
        """
        normalized_site, transformation = NormalizeSite.NormalizeSite(self.pdb_lines, site_name, site_simplification)
        normalized_molecule = NormalizeMolecule.NormalizeMolecule(path_to_sdf)
        self.CheckExtract(normalized_site, normalized_molecule)
        
        molecule_moves = NeuralDocking.GetMoleculeMoves(normalized_site, normalized_molecule)
        normalized_molecule.MoveMolecule(molecule_moves, transformation)
        normalized_site.MoveSite(transformation)
        
        if energy_normalization:
            normalized_molecule.NormalizeEnergy(normalized_site)
        if post_translation:
            normalized_molecule.PostTranslate(normalized_site)
        
        path_to_result = normalized_molecule.ExportPath(export=True)
        results = NDockResults(normalized_site, normalized_molecule)
        
        return path_to_result, results
    
    def CheckExtract(self, site, molecule):
        """
            Helper function to check and bactrack irregularities and mistakes present in both normalized site and ligand (here under name               molecule). This function is to expand, as possible errors that could arise (mainly thanks to varience of pdb files) are still               being found and accounted for.
        
            Parameters:
                site: SiteAmino class, contains list of all amino acids and their properties present in the protein's site.
                molecule: MoleculeAmino, contains information to fully reconstruct molecule, list of atoms, bonds and more
        
            Returns:
                Nothing, if no error is present. If some irregularity is found, error message, in the form of sys, is returned to user.
            """
        for amino in site.amino:
            if len(amino.residues)==0:
                sys.exit("There was a problem in extracting atoms for residue: " + print(amino))
            
        if len(molecule.atoms) <= 0:
            sys.exit("There are no atoms present in the molecule.")
            
        if len(molecule.bonds) <= 0:
            sys.exit("There are no bonds present in the molecule.")
    
    
class NDockResults:
    """
    NDockResults is a class containing resulting class from the NDock operations. It serves to 
    perform additional post-docking operations, such as charge (under tha mantle of RDKit)
    or score (Free energy estimation, constants from Autodock4) computations.
    
     Functions:
         __init__(self, site, molecule): 
             Assigns the values of molecule and site, which were taken fromNDock computations. Can 
             also be initiated by user, if molecule and protein are given in rightclass format 
             (SiteAmino, MoleculeAmino classes).
        ElectricCharges(self): 
            Doesn't take an input. It firstly exports the site and molecule as pdb and sdf file respectively,
            then it uses RDKit to compute the given electrostatic charges which it outputs in lists, by 
            atom order.
            
    Attributes:
        self.site: A class SiteAmino, normally exported class from NDock, but can be created by user.
            Contains informations about all amino acids in the site of the protein.
        self.molecule: A class MoleculeAmino, normally exported from NDock, but can be created by user.
            Constains moleucle name, its bonds and atoms.
    """
    def __init__(self, site, molecule):
        """
        This function initiates the NDockResults class by taking both the SiteAmino and MoleculeAmino classes. These classes are normaly             returns from the NDoc.Dock function, but can also be created by the user.
        
        Parameters:
            site: SiteAmino class, generally result from the NDock.Dock function.
            molecule: MoleculeAmino, generally result from the NDock.Dock function.
        
        Returns:
            None
        """
        self.site = site
        self.molecule = molecule
        
    def compute_atom_distance(self, atom1_distances, atom2_distances):
        """
        Auxilary function of the ElectricCharges function, present in the same class. Takes as input lists of atom positions and then               computes euclidian distance between both of them, units are intended to be angstorms, but depends on unit inputed.
        
        Parameters:
            atom1_distances: List of floats, bears the atom's coordinates in space (x, y, z).
            atom2_distances: Same as atom1_distances, just other atom.
        
        Returns:
            distance: Float, gives the euclidian distance in space between two specified atoms, in units given.
        """
        x1, y1, z1 = atom1_distances
        x2, y2, z2 = atom2_distances

        x_component = (x1-x2)**2
        y_component = (y1-y2)**2
        z_component = (z1-z2)**2
        
        distance = np.sqrt(x_component+y_component+z_component)

        return distance
        
    def ElectricCharges(self):
        """
        Is completely dependable on functions taken from RDKit package. Its purpose is to compute electric charges both for the active site         and for the ligand. It firstly exports both the molecule and the site in form of sdf and pdb files, respectively. Then they are             combined and embedded, extracting the computed charges.
        
        Parameters:
            There are no additional parameters, all are taken from the class parameters.
        
        Returns:
            molecule_charges: List of floats, this list containes list of charges in the same order as is the order of atoms in the                         self.molecule MoleculeAmino class.
            site_charges: List of floats, this list containes list of charges in the same order as is the order of residues and their atoms                 in the self.site SiteAmino class.
        """
        path_molecule = self.molecule.ExportPath()
        path_site = self.site.ExportPath()
        
        molecule = Chem.MolFromMolFile(path_molecule)
        site_path = open(path_site, 'r').read()
        site = Chem.MolFromPDBBlock(site_path)
        combined = Chem.CombineMols(molecule, site)
        
        combined_hs = Chem.AddHs(combined)
        AllChem.EmbedMolecule(combined_hs)
        _, result = rdEHTTools.RunMol(combined_hs)
        static_charges = result.GetAtomicCharges()[:combined.GetNumAtoms()]
        
        molecule_charges = static_charges[0:len(self.molecule.atoms)]
        site_charges = static_charges[len(self.molecule.atoms):len(self.molecule.atoms)+self.site.NumAtoms()]
        
        return molecule_charges, site_charges
    
    def Score(self, soft_docking=1, solvation = True):
        """
        This function uses physical equation which combines Lenard-Jones Potential, electrostatic potentials and, if wished by the user,             also solvation to find estimation of free energy of ligand-site binding. For better, converging result, additional constants are             used. These are taken from the Autodock software. Soft docking, acounting for dynamic active site, is available, but set to zero             initially. Docking score serves just as rough estimate, and it should be taken lightly, only as general guide to the results. It can         sometimes happen that the true result has score in hundreds of Joules, but these are rare cases.
        
        Parameters:
            soft_docking=1: Float, number used for soft docking, should be between 0 and 1, the smaller the soft_docking is, the more                       dynamic site is accounted. The correct number depends on situation but good start is around 0.7-0.8.
            solvation=True: Bool, says function whether to account for solvation, it should be used in sites where lot of water is present                   before docking, but not so much in hydrophobic ones, as it then skews the energy estimation.
        
        Returns:
            score: Float, represents the estimation of the free energy tahat is released during the ligand-site binding process, the more                   negative it is, the better the result should be (it is only rough estimate, so be beware of inconsistensies, final judgement                 should always rest on user).
        """
        molecule_atoms = self.molecule.atoms
        site_atoms = []
        for residue in self.site.amino:
            site_atoms+=residue.residues
        score = 0
        
        molecule_charges, site_charges = self.ElectricCharges()
        
        for i,site_atom in enumerate(site_atoms):
            for j,molecule_atom in enumerate(molecule_atoms):
        
                epsilon_ij = Well_depth[molecule_atom.name][site_atom.name] #Well depth of both ligand and active site atoms
                sigma_ij = Walls_Radiuses[molecule_atom.name]+Walls_Radiuses[site_atom.name] #Combination of both Walls radiuses

                charge_i = molecule_charges[j]
                charge_j = site_charges[i]
            
                r_ij=self.compute_atom_distance([molecule_atom.coord_x,molecule_atom.coord_y,molecule_atom.coord_z],[site_atom.coord_x,site_atom.coord_y,site_atom.coord_z])+1e-10
            
                Hi = constants["e0"]-constants["F"]
            
                eps_func = constants['F']+(Hi/(constants["lambda"]+constants["k"]*np.exp(-constants["l"]*Hi*r_ij)))

                dG_elec = (charge_i*charge_j)/(r_ij*eps_func)

                A_ij = (sigma_ij**12)*epsilon_ij
                B_ij = 2*(sigma_ij**6)*epsilon_ij

                dG_Waals = soft_docking*((A_ij/(r_ij**12))-(B_ij/(r_ij**6)))
                
                if solvation==True:
                    Volume_i = (4/3)*np.pi*Atomic_radiuses[molecule_atom.name]**3
                    Volume_j = (4/3)*np.pi*Atomic_radiuses[site_atom.name]**3
                    S_i = Solvation[molecule_atom.name]+0.01097*charge_i
                    S_j = Solvation[site_atom.name]+0.01097*charge_j
                    dG_solvate = (Volume_i*S_j+Volume_j*S_i)*np.exp(-r_ij**2/(2*sigma_ij**2))
                else:
                    dG_solvate = 0
            
                score += 0.1485*dG_Waals + 0.1146*dG_elec + 0.1711*dG_solvate #Constants for better results, adopted from Autodock
        
        return score