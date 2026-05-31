import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from pathlib import Path
import sys

# ==========================================
# 1. RICERCA ROOT DINAMICA
# ==========================================
def get_project_root(current_path: Path) -> Path:
    for parent in [current_path, *current_path.parents]:
        if (parent / ".env").exists():
            return parent
    raise FileNotFoundError("Impossibile trovare il file .env per stabilire la root.")

PROJECT_ROOT = get_project_root(Path(__file__).resolve())

# Configurazione Path
# Assicurati che il file JSON sia nella cartella root del progetto
CREDENTIALS_PATH = PROJECT_ROOT / "credentials.json"
SHEET_NAME = "Ai4CyberProject-Group16" # Cambia se hai dato un nome diverso al foglio

class GoogleSheetLogger:
    def __init__(self):
        self.scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        self.sheet = None

        try:
            if not CREDENTIALS_PATH.exists():
                print(f"[ERROR Logger] File credenziali non trovato in: {CREDENTIALS_PATH}")
                return

            creds = Credentials.from_service_account_file(str(CREDENTIALS_PATH), scopes=self.scope)
            client = gspread.authorize(creds)
            self.sheet = client.open(SHEET_NAME).sheet1
            print(f"[SUCCESS Logger] Connesso al foglio Google: '{SHEET_NAME}'")
            
        except Exception as e:
            print(f"[ERROR Logger] Connessione a Google Sheets fallita: {e}")

    def log_biometric_metrics(self, tester, phase, attack_type, epsilon, defense_type, 
                              accuracy, eer, far, frr, threshold, notes=""):
        """
        Invia i dati della Traccia 1 al foglio Google.
        Ordine colonne:
        A: Timestamp, B: Tester, C: Fase_Progetto, D: Tipo_Attacco, E: Epsilon,
        F: Difesa, G: Accuracy, H: EER, I: FAR, J: FRR, K: Soglia_Ottimale, L: Note
        """
        if self.sheet is None:
            print("[WARNING] Logger non connesso. Metriche non salvate su cloud.")
            return

        try:
            # Formattiamo i valori float per avere 4 cifre decimali ed evitare formattazioni strane
            row = [
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                tester,
                phase,               # es. "Clean Evaluation", "Targeted Attack"
                attack_type,         # es. "None", "FGSM", "PGD"
                str(epsilon),        # es. "0", "0.05", "0.1"
                defense_type,        # es. "None", "JPEG Compression"
                f"{accuracy:.4%}".replace('.', ','),  # Formattazione ITA per Google Sheets
                f"{eer:.4%}".replace('.', ','),
                f"{far:.4%}".replace('.', ','),
                f"{frr:.4%}".replace('.', ','),
                f"{threshold:.4f}".replace('.', ','),
                notes
            ]
            
            self.sheet.append_row(row, value_input_option='USER_ENTERED')
            print(f"[INFO] Metriche loggate correttamente per il tester: {tester} [Eps: {epsilon}]")
            
        except Exception as e:
            print(f"[ERROR Logger] Errore durante l'invio al foglio: {e}")

# ==========================================
# TEST DI FUNZIONAMENTO Esempio
# ==========================================
if __name__ == "__main__":
    logger = GoogleSheetLogger()
    
    # Esempio di riga che verrà scritta
    logger.log_biometric_metrics(
        tester="Francesco",
        phase="Test Logger",
        attack_type="None",
        epsilon=0.0,
        defense_type="None",
        accuracy=0.0,
        eer=0.0,
        far=0.0,
        frr=0.0,
        threshold=0.0,
        notes="Test di connessione API"
    )