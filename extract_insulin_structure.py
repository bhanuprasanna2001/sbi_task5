from Bio.PDB import PDBList, PDBParser
from Bio.SeqUtils import seq1

# Step 1: Download PDB file
pdb_id = "1A7F"
pdbl = PDBList()
pdbl.retrieve_pdb_file(pdb_id, pdir=".", file_format="pdb")  # Saves as pdb1a7f.ent

# Step 2: Extract sequence from Chain A
parser = PDBParser(QUIET=True)
structure = parser.get_structure("insulin", "pdb1a7f.ent")

residues = []
res_ids = []

for model in structure:
    for chain in model:
        if chain.id == "A":
            for residue in chain:
                if residue.id[0] == " ":
                    try:
                        aa = seq1(residue.resname)
                        residues.append(aa)
                        res_ids.append(residue.id[1])
                    except KeyError:
                        continue

sequence = ''.join(residues)

# Step 3: Parse HELIX regions from the raw file
helix_ranges = []
with open("pdb1a7f.ent") as f:
    for line in f:
        if line.startswith("HELIX") and line[19] == 'A':
            start = int(line[21:25].strip())
            end = int(line[33:37].strip())
            helix_ranges.append((start, end))

# Step 4: Label each residue as 'H' or 'O'
labels = []
for res_id in res_ids:
    label = 'O'  # default = other
    for start, end in helix_ranges:
        if start <= res_id <= end:
            label = 'H'
            break
    labels.append(label)

# Final output
print("\n✅ Amino Acid Sequence (Chain A):")
print(sequence)

print("\n🔁 Secondary Structure Labels:")
print(''.join(labels))

structure = parser.get_structure("insulin", "pdb1a7f.ent")

import json

# Convert amino acids to indices
aa_to_index = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9,
    'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
}
input_indices = [aa_to_index[aa] for aa in sequence]

# Convert 'H'/'O' labels to 1/0
label_indices = [1 if l == 'H' else 0 for l in labels]

# Save as JSON for BayesFlow testing
sample = {
    "input": input_indices,
    "target": label_indices
}

with open("insulin_test_sample.json", "w") as f:
    json.dump(sample, f, indent=2)

print("\n✅ Saved insulin test sample to: insulin_test_sample.json")
