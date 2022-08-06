#Encoder used in converting Selfies molecular representationto one used by neural networks
MoleculeEncoder = {'[Zn]': 0, '[N]': 1, '[C]': 2, '[=Branch1]': 3, '[=O]': 4, '[=C]': 5, '[N+1]': 6, 
                   '[Branch2]': 7, '[Branch1]': 8, '[S]': 9, '[O]': 10, '[Ring2]': 11, '[P]': 12, 
                   '[=N]': 13, '[Ring1]': 14, '[#Branch1]': 15, '[#Branch2]': 16, '[=Branch2]': 17, 
                   '[F]': 18, '[=S]': 19, '[Ca]': 20, '[Mg]': 21, '[C@H1]': 22, '[P@@]': 23, 
                   '[O-1]': 24, '[C@@H1]': 25, '[P@]': 26, '[NH1]': 27, '[/C]': 28, '[\\C]': 29, 
                   '[\\C@@H1]': 30, '[C@]': 31, '[#C]': 32, '[NH2]': 33, '[=P]': 34, '[Mn]': 35, 
                   '[NH3+1]': 36, '[Na]': 37, '[=Ring1]': 38, '[=N+1]': 39, '[K]': 40, '[N-1]': 41, 
                   '[=Ring2]': 42, '.': 43, '[Fe+2]': 44, '[Co]': 45, '[C@@]': 46, '[/F]': 47, 
                   '[#N]': 48, '[Fe]': 49, '[Cl]': 50, '[Cu]': 51, '[/N]': 52, '[S+1]': 53, 
                   '[\\S]': 54, '[P+1]': 55, '[S-1]': 56, '[Mo]': 57, '[I]': 58, '[Mo+2]': 59, 
                   '[/O]': 60, '[V]': 61}
#Conversion of amino acid representation for neuralnetworks
AminoEncoder = {"ala":0,"arg":1,"asn":2,"asp":3,"cys":4,"glu":5,"gln":6,"gly":7,"his":8,"ile":9,
                "leu":10,"lysa":11,"met":12,"phe":13,"pro":14,"ser":15,"thr":16,"trp":17,"tyr":18,
                "val":19}
#Conversion of atoms for neuarl sites
atomnumber = {'o':0,'c':1,'h':2,'n':3,'ca':3,'p':4,'k':5,'s':6,'na':7,'cl':8,'mg':9,'b':10,
              'cr':11,'co':12,'cu':13,'f':14,'i':15,'fe':16,'mn':17,'mo':18,'se':19,
              'si':20,'sn':21,'v':22,'zn':23}
#List of acceptable atoms by the program
atom_check = ['o','c','h','n','ca','p','k','s','na','cl','mg','b','cr','co','cu','f','i',
              'fe','mn','mo','se','si','sn','v','zn']
#List of acceptable amino acids by the program
amino_acids = ["ala","arg","asn","asp","cys","glu","gln","gly","his","ile","leu","lysa",
              "met","phe","pro","ser","thr","trp","tyr","val"]
#Walls radiuses for free energy computation
Walls_Radiuses = {'h':1.0,'he':1.4,'li':1.81,'be':1.53,'b':1.92,'c':2,'n':1.75,'o':1.6,'f':1.47,'ne':1.54,'na':2.27,'mg':1.73,'al':1.84,'si':2.1,'p':1.8,'s':2,'cl':1.75,'ar':1.88,'k':2.75,'ca':2.31,'sc':2.11,'ti':2.46,'v':2.42,'cr':2.45,'mn':2.45,'fe':2.44,'co':2.4,'ni':2.4,                 'cu':1.4,'zn':1.39,'ga':1.87,'ge':2.11,'as':1.85,'se':1.9,'br':1.85,'kr':2.02,'rb':3.03,'sr':2.49,'ru':2.46,'pd':1.63,'ag':1.72,'cd':1.58,'in':1.93,'sn':2.17,'sb':2.06,'te':2.06,'i':1.98,'xe':2.16,'cs':3.43,'ba':2.68,'sm':2.87,'os':2.41,'u':1.86}
#Atomic radiuses for free energy computation
Atomic_radiuses = {'h':0.53,'c':0.67,'n':0.56,'s':0.88,'o':0.48}
#Well depthfor free energy computation
Well_depth = {'c':{'c':0.150,'n':0.155,'o':0.173,'s':0.173,'h':0.055},'n':{'c':0.155,'n':0.160,'o':0.179,'s':0.179,'h':0.057},'o':{'c':0.173,'n':0.179,'o':0.200,'s':0.200,'h':0.063},'s':{'c':0.173,'n':0.179,'o':0.200,'s':0.200,'h':0.063}, 'h':{'c':0.055,'n':0.057,'o':0.063,'s':0.063,'h':0.020}}
#Special values for free energy computation (Electrostatic parameters)
C12 = {'o':75570,'n':75570,'s':2657200,'c':0,'h':0}
C10 = {'o':23850,'n':23850,'s':354290,'c':0,'h':0}
#Parameters for solvation for free energy computation
Solvation = {'c':-0.00143,'n':-0.00162,'o':-0.00251,'s':-0.00214,'h':0.00051}
#Special constants for free energy computation
constants = {"l":0.003627, "k":7.7839, "F":-8.5525, "lambda":3.5, "e0":78.4, "a":332}