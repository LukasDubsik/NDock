import sys
sys.path.append('..')
from Classes import Site, Residue, AminoAcid, SiteAmino, AminoAtom, Transformation
from Constants import amino_acids


def FindSite(site_name, pdb_lines):
    
    site = Site(None, None, None, None)
    
    run_residues = []
    for i,line in enumerate(pdb_lines):
        if line[0:4].lower()=="site" and line[11:14].lower()==site_name:
            if line[7:10].strip()=="1":
                site.name = site_name
                site.num_res = line[15:17].strip()
            if pdb_lines[i+1][11:14].lower()!=site_name:
                site.num_lin = line[7:10].strip()
            for k in range(18,52,11):
                name = line[k:k+3].strip().lower()
                if name != "" and name!="hoh" and name in amino_acids:
                    run_residues.append(Residue(line[k:k+3].strip().lower(), line[k+4].strip().lower(), line[k+5:k+9].strip().lower()))
             
    if site.name!=None and site.num_res!=None and site.num_lin!=None and len(run_residues)>1:
        site.residues = run_residues
        return site
    else:
        return None

def ExtractSite(pdb_lines, site):
    
    current_res = site.residues[0]
    amino = AminoAcid(current_res.res_name, current_res.chain, current_res.res)
    pre_site = SiteAmino()
    counter = 1
    
    for i,line in enumerate(pdb_lines):
        if line[0:6].strip().lower() == "atom":
            if line[17:20].strip().lower()==current_res.res_name and line[21].strip().lower()==current_res.chain and line[22:26].strip().lower()==current_res.res:
                atom = AminoAtom(line[12:16].strip().lower(),float(line[30:38]),float(line[38:46]),float(line[46:54]), line[76:78].strip().lower())
                amino.add(atom)  
                
            run_res = site.residues[counter]
                
            if pdb_lines[i+1][17:20].strip().lower()==run_res.res_name and pdb_lines[i+1][21].strip().lower()==run_res.chain and pdb_lines[i+1][22:26].strip().lower()==run_res.res:
                if counter != len(site.residues)-1:
                    pre_site.add(amino)
                    current_res = run_res
                    amino = AminoAcid(current_res.res_name, current_res.chain, current_res.res)
                    counter += 1
                elif current_res != run_res:
                    current_res = run_res
                    pre_site.add(amino)
                    amino = AminoAcid(current_res.res_name, current_res.chain, current_res.res)
                else:
                    current_res = run_res
    pre_site.add(amino)
    return pre_site

def NormalizeSite(pdb_lines, site_name, site_simplification):
    if isinstance(site_name, str):
        site = FindSite(site_name.lower().strip(), pdb_lines)
    elif isinstance(site_name, list):
        site = Site(None, None, len(site_name), site_name)
    else:
        sys.exit("The variable site_name is neither string or list")
    if site==None:
        sys.exit("Sorry, but the given site, " + site_name + " , could not be extracted.")    
    extracted_site = ExtractSite(pdb_lines, site)
    
    m_x, m_y, m_z = extracted_site.GetMean(transform=True)
    alpha, beta, gamma, alpha2, beta2, gamma2 = extracted_site.GetRotation(transform=True)
    
    return extracted_site, Transformation(m_x, m_y, m_z, alpha, beta, gamma, alpha2, beta2, gamma2)
