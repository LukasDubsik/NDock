# Notice
The initial, reworked version is now available. Soon, download will be available through conda.

# NDock
NDock is a bioinformatic library capable of molecular docking using neural networks. It is a part of larger library, under the name of BIP (Bioinformatic Python), currently in development. 

## Use
__NDock__ aims to be alternative to other docking tools, such as softwares GOLD or AutoDock. While these softwares use probabilistic algorithms, NDock relies on neural networks as its primary tool. Additional precission is then provided by energy constraints minimalization and guided strain removal. Ndock builds on simplicity of use and short runtime. As in other softwares user needs substantial knowledge in bionfarmatics, NDock is easy to run with just few clicks without any prior knowledge of the field, the use of neural networks also provides quick speed of execution. 

## Evaluation
To evaluate NDock, comparison was performed with program Autodock Viva. 20 randomly selected proteins, none of them were previously parts of training dataset, with theier active sites and belonging ligands. Site simplification was applied, but energy normalization or post translation were not. The resulting RMSD error was: __1.5784__ for Autodock and __1.8562__ for NDock. But NDock completed this task 10x times faster (not taking into account the preparation required by Viva)!

NDock is ***still in its early phases of development***, so it has yet to reach its full potential. There may still be some bugs or errors and the precision is still lacking. If any problems are encountred or some suggestions are to be given, please do not hesitate to contact me.

## Library
Current extent of main fuctions of NDock library as of this update:

```python
from NDock import Main

ndock = Main.NDock(path_to_pdb) # Creates class of NDock with one protein
path_to_result, results = ndock.Dock(path_to_sdf, site_name, site_simplification=False, energy_normalization=False,
                            post_translation=False) # performs docking on molecule and specified site
charges_molecule, charges_site = results.ElectricCharges() # Returns electrostatic charges of individual atoms, relies on RDKit
score = results.Score(soft_docking = 1, solvation = True) # Accesses the score of finalised docking (docking score)
```

The Library is then composed of two main classes, the class NDock itself plus class holding its results, the NDockReturn class. Both are described below.
### class NDock
This is the description of the main class NDock.

```python
class NDock:
    """
    NDock class serves to dock ligands (small molecules reacting with proteins) into specified 
    proteins. It relies on neural sites as its mean of achieving this. In use are also energy 
    normalization and strain removal. 
    
    Functions:
        __init__(self, path_to_pdb): Initiates the class. It takes path to protein file (currently only pdb, 
            soon mmcif), from which it then excrates lines with whom further functions work.
        Dock(self, path_to_sdf, site_name, site_simplification=False, energy_normalization=True,post_translation=True): 
            Takes as input path to sdf (molecule to be docked) file and site name of previously specified protein.
            This function performs the docking itself. It firstly normalizes input, then uses neural site and, if 
            specified, another functionalities.
            
    Attributes:
        self.path_to_pdb: A string indicating path to specified pdb file.
        self.pdb_lines: A list containing the extracted lines of the pdb file.
    """
```
### class NDockReturn
This is the description of the supporting class NDockResults.

```python
class NDockResults:
    """
    NDockResults is a class containing resulting class from the NDock operations. It serves to 
    perform additional post-docking operations, such as charge (under tha mantle of RDKit)
    or score (Free energy estimation, constants from Autodock4) computations.
    
    Functions:
         __init__(self, site, molecule): 
             Assigns the values of molecule and site, which were taken fromNDock computations. Can 
             also be initiated by user, if molecule and protein are given in right class format 
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
```
## Future development
The next, foreseeable development consists of: 
- [ ] Finishing commenting all parts of the code (currently full comments available only in main functions).
- [ ] Expanding available functions in NDockResults class.
- [ ] Fixing the post translation function, which currently does not work as result of errors in orbital overlap calculations. 
- [ ] Adding the support for mmcif files of proteins and other files for molecules.
- [ ] Adding full reader and writer class for pdb and sdf files for easier preparation of site and molecule by the user.
- [ ] Removing reliance on RDKit functions by adding own functions (currently in progress) and migrating those functions to C++.
And of course removal of possible bugs and general betterment of program precision.
More in depth informations will be available in ensuing research paper, link to which will be given here.

### Main libraries used
- RDKit
- Numpy
- Pytorch
- Selfies

### Scientific papers references
__Autodock__:
- Eberhardt, J., Santos-Martins, D., Tillack, A.F., Forli, S. (2021). AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings. Journal of Chemical Information and Modeling.
- Trott, O., & Olson, A. J. (2010). AutoDock Vina: improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. Journal of computational chemistry, 31(2), 455-461.

__RDKit__:
- RDKit: Open-source cheminformatics; http://www.rdkit.org
