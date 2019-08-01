import torch
from pymatgen import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import schnetpack as spk
from ase.db import connect

best_model = torch.load('best_model', map_location='cpu')

structures = []
structure = Structure.from_file('input.cif', primitive=True)
structures.append(AseAtomsAdaptor.get_atoms(structure))
structures.append(AseAtomsAdaptor.get_atoms(structure))
structures.append(AseAtomsAdaptor.get_atoms(structure))

with connect('cifs.db') as con:
    for i, at in enumerate(structures):
        con.write(at)

dataset = spk.AtomsData('cifs.db')
dataset_loader = spk.AtomsLoader(dataset, batch_size=2)

device = torch.device('cpu')
for count, batch in enumerate(dataset_loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    pred = best_model(batch)
    print(pred)
