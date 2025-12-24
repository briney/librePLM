import numpy as np
import torch
from Bio.PDB import PDBIO, Atom, Chain, Model, Residue, Structure
from Bio.SeqUtils import seq3


def build_pdb(
    coordinates: np.ndarray | torch.Tensor,
    output_file: str,
    sequence: str | None = None,
) -> None:
    """
    Builds a PDB file from a protein sequence and backbone coordinates.
    If the sequence is not provided, only the backbone coordinates will be used
    to construct the PDB file (with all residues set to 'UNK').

    Args:
    - coordinates (np.ndarray | torch.Tensor): Shape (L, 3, 3) for N, CA, C atoms per residue. Required.
    - output_file (str): Path to save the PDB file. Required.
    - sequence (str): One-letter amino acid sequence (e.g., 'ACDE'). If not provided, all residues will be set to 'UNK'.
    """
    # check sequence input
    if sequence is None:
        sequence = "X" * coordinates.shape[0]
    if len(sequence) != coordinates.shape[0]:
        raise ValueError(
            "Sequence length must match number of residues in coordinates."
        )
    # convert coordinates to numpy (required for Bio.PDB)
    if isinstance(coordinates, torch.Tensor):
        coordinates = coordinates.numpy()

    # initialize the structure, model and chain
    structure = Structure.Structure("protein")
    model = Model.Model(0)
    chain = Chain.Chain("A")

    atom_serial = 1  # start atom numbering from 1
    for i, aa in enumerate(sequence):
        # convert 1-letter to 3-letter code (e.g., 'A' -> 'ALA')
        res_name = seq3(aa).upper()
        # create residue (hetero flag ' ', resseq i+1, insertion code ' ')
        res = Residue.Residue(id=(" ", i + 1, " "), resname=res_name, segid=" ")
        # add atoms: N, CA, C
        for j, atom_name in enumerate(["N", "CA", "C"]):
            coord = coordinates[i, j]  # [x, y, z]
            if not isinstance(coord, np.ndarray):
                coord = np.array(coord, dtype=float)

            atom = Atom.Atom(
                name=atom_name,
                coord=coord,
                bfactor=0.0,  # b-factor (can maybe use for pLDDT or something in the future)
                occupancy=1.0,
                altloc=" ",
                fullname=f" {atom_name.ljust(3)}",  # full atom name (CA, N, C)
                serial_number=atom_serial,  # unique atom ID
                element=atom_name[0],  # element symbol (CA -> C)
            )
            res.add(atom)
            atom_serial += 1

        chain.add(res)

    model.add(chain)
    structure.add(model)

    # save to PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)
