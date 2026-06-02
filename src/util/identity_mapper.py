import pandas as pd
from pathlib import Path

class IdentityMapper:
    """
    Classe utility per mappare gli output della rete FaceNet (0-8630) 
    alle identità reali di VGGFace2 leggendo direttamente identity_meta.csv.
    """
    def __init__(self, meta_csv_path: str | Path):
        self.meta_csv_path = Path(meta_csv_path)
        
        if not self.meta_csv_path.exists():
            raise FileNotFoundError(f"Impossibile trovare {self.meta_csv_path}")
            
        # 1. Carichiamo il dataset ignorando gli spazi nei nomi delle colonne
        self.df_meta = pd.read_csv(self.meta_csv_path, skipinitialspace=True)
        
        # 2. Pulizia: Togliamo virgolette e spazi dalle stringhe
        self.df_meta['Class_ID'] = self.df_meta['Class_ID'].astype(str).str.strip()
        self.df_meta['Name'] = self.df_meta['Name'].astype(str).str.replace('"', '').str.strip()
        
        # 3. IL FILTRO D'ORO: Teniamo SOLO le classi usate nel Pre-Training (Flag == 1)
        self.df_train = self.df_meta[self.df_meta['Flag'] == 1].copy()
        
        # 4. Resettiamo l'indice. Ora l'indice (0-8630) corrisponde esattamente all'output di FaceNet!
        self.df_train.reset_index(drop=True, inplace=True)
        
        # 5. Creiamo dizionari per ricerche istantanee (O(1) complexity)
        # Dizionario: facenet_id (es. 0) -> Dizionario info
        self.id_to_info = self.df_train.to_dict('index')
        
        # Dizionario: Class_ID (es. 'n000002') -> facenet_id (es. 0)
        self.class_id_to_facenet_id = {row['Class_ID']: idx for idx, row in self.df_train.iterrows()}
        
    def get_info_by_facenet_id(self, facenet_id: int) -> dict:
        """Passi l'output della rete (es. 5301), ritorna le info dell'identità."""
        if facenet_id in self.id_to_info:
            return self.id_to_info[facenet_id]
        return None

    def get_facenet_id_by_class_id(self, class_id: str) -> int:
        """Passi l'ID (es. 'n007726'), ritorna l'output atteso della rete. Ritorna -1 se Flag==0."""
        return self.class_id_to_facenet_id.get(class_id.strip(), -1)
        
    def is_valid_for_attack(self, class_id: str) -> bool:
        """Controlla se l'identità faceva parte del pre-training."""
        return self.get_facenet_id_by_class_id(class_id) != -1

    def get_num_training_classes(self) -> int:
        """Dovrebbe ritornare esattamente 8631."""
        return len(self.df_train)