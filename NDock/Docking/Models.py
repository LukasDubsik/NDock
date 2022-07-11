import torch
import torch.nn as nn
import torch.nn.functional as F

class NDocker(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = encoder
        
        self.amino0 = nn.Linear(26, 256)
        self.amino1 = nn.Linear(256, 512)
        self.site_eval = nn.Linear(512, 512)
        self.molecule = nn.Linear(512, 512)
        self.lin1 = nn.Linear(1024, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, 256)
        self.lin4 = nn.Linear(256, 6)
        
        self.drop = nn.Dropout(0.7) 
        
    def forward(self, molecule, site):
        
        hidden_molecule = self.encoder.init_hidden()
        for atom in molecule:
            hidden_molecule = self.encoder(atom.to(self.device), hidden_molecule)
        hidden_molecule = self.drop(F.leaky_relu(self.molecule(hidden_molecule))).squeeze(0)
        
        for i,residue in enumerate(site):
            residue = residue.to(self.device)
            res = self.drop(F.leaky_relu(self.amino0(residue.unsqueeze(0).unsqueeze(0))))
            res = self.drop(F.leaky_relu(self.amino1(res)))
            res = res.squeeze(-1)
        run_eval = self.drop(F.leaky_relu(self.site_eval(res))).squeeze(0)
        
        inp = torch.cat((run_eval, hidden_molecule), dim=1).squeeze(1)
        
        lin = self.drop(F.leaky_relu(self.lin1(inp)))
        lin = self.drop(F.leaky_relu(self.lin2(lin)))
        lin = self.drop(torch.sigmoid(self.lin3(lin)))
        results = torch.sigmoid(self.lin4(lin))
            
        return results
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hid_dim = 512
        
        self.lin = nn.Linear(62, 512)
        self.gru = nn.GRU(512, 512, batch_first=True)
        
        self.drop = nn.Dropout(0.7)
    
    def forward(self, data, hidden):
        lin = self.drop(F.leaky_relu(self.lin(data)))
        out, hid = self.gru(lin.unsqueeze(0).unsqueeze(0), hidden)
        hid = self.drop(hid)
        
        return hid
    
    def init_hidden(self):
        
        weight = next(self.parameters()).data
        
        if (self.device):
            hidden = weight.new(1, 1, self.hid_dim).zero_().cuda()
        else:
            hidden = weight.new(1, 1, self.hid_dim).zero_()
        
        return hidden