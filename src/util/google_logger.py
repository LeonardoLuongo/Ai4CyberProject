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

    def log_attack_metrics(self, tester, attack_type, strategy, epsilon, defense_type, 
                           robust_accuracy, targeted_asr, untargeted_asr, notes=""):
        """
        Invia i dati della Traccia 1 al foglio Google.
        Colonne: 
        A: Timestamp, 
        B: Tester, 
        C: Tipo_Attacco, 
        D: Strategia, 
        E: Epsilon, 
        F: Difesa, 
        G: Robust_Accuracy, 
        H: Targeted_ASR, 
        I: Untargeted_ASR, 
        J: Note
        """
        if self.sheet is None:
            print("[WARNING] Logger non connesso. Metriche non salvate su cloud.")
            return

        try:
            row = [
                datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                tester,
                attack_type,         # es. "PGD"
                strategy,            # es. "least_likely" o "untargeted"
                f"{float(epsilon):.4f}".replace('.', ','),
                defense_type,        # es. "None"
                f"{robust_accuracy:.4%}".replace('.', ','), 
                f"{targeted_asr:.4%}".replace('.', ','),
                f"{untargeted_asr:.4%}".replace('.', ','),
                notes
            ]
            
            self.sheet.append_row(row, value_input_option='USER_ENTERED')
            print(f"[INFO Logger] Inviato a Google Sheets: {attack_type} | Eps: {epsilon} | Strat: {strategy}")
            
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