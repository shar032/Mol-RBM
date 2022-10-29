""" Cheminformatics utilities for RBM to enable comparison 
    of seed input and generated SMILES and general purposes """
    
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

class Chem_utils:
    def check_SMILES(self, s):
        try:
            Chem.MolFromSmiles(s)
            return True
        except:
            return False
        
    def get_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return mol
    
    def get_smiles(self, mol):
        smiles = Chem.MolToSmiles(mol)
        return smiles
    
    def get_rdk_fingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = Chem.RDKFingerprint(mol)
        return fp
    
    def get_fingerprint_similarity_pair(self, fp1, fp2):
        sim = DataStructs.FingerprintSimilarity(fp1,fp2)
        return sim
    
    def get_morgan_fingerprint(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprint(mol,2)
        return fp
    
    def get_dice_similarity(self, fp1, fp2):
        sim = DataStructs.DiceSimilarity(fp1, fp2)
        return sim
    
    def get_atomic_num_set_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        atom_list = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        return set(atom_list)
    
    def get_num_heavy_atom_count(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        return Chem.Lipinski.HeavyAtomCount(mol)
    
    def get_molar_mass(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        molar_mass = Descriptors.MolWt(mol)
        return molar_mass
    
    