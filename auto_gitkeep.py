import os

print("Ricerca cartelle vuote in corso...")

for dirpath, dirnames, filenames in os.walk('.'):
    # Escludiamo la cartella .git per non corrompere il repository
    if '.git' in dirnames:
        dirnames.remove('.git')
    
    # Se la cartella non ha né sottocartelle né file, è vuota
    if not dirnames and not filenames:
        filepath = os.path.join(dirpath, '.gitkeep')
        with open(filepath, 'w') as f:
            pass
        print(f"[CREATO] {filepath}")

print("Operazione completata!")