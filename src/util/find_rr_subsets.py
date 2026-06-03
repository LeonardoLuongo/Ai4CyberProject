import os
import cv2
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans

from facenet_pytorch import InceptionResnetV1, MTCNN
from identity_mapper import IdentityMapper

def main():
    print("======================================================")
    print(" ROUND-ROBIN STRATEGY: CALCOLO DEI 3 SUBSET LOGICI    ")
    print("======================================================\n")

    base_dir = Path(os.getcwd())
    csv_path = base_dir / "dataset" / "clean" / "splits" / "manifest.csv"
    meta_path = base_dir / "dataset" / "clean" / "splits" / "identity_meta.csv"
    out_json = base_dir / "dataset" / "clean" / "splits" / "rr_subsets.json"

    mapper = IdentityMapper(meta_csv_path=meta_path)
    df = pd.read_csv(csv_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, post_process=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    identities = sorted(df['identity_id'].unique())
    
    centroids = {}
    confidences = {}

    print("-> Calcolo Centroidi 512D e Confidenze...")
    with torch.no_grad():
        for ident in tqdm(identities):
            facenet_id = mapper.get_facenet_id_by_class_id(ident)
            if facenet_id == -1: continue

            subset = df[df['identity_id'] == ident]
            embs, confs = [], []

            for _, row in subset.iterrows():
                img_bgr = cv2.imread(str(base_dir / row['image_path']))
                if img_bgr is None: continue
                t_crop = mtcnn(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                if t_crop is None: continue

                t_img = t_crop.unsqueeze(0).to(device)
                
                # Confidenza
                resnet.classify = True
                logits = resnet(t_img)
                if int(torch.argmax(logits)) == facenet_id:
                    confs.append(float(F.softmax(logits, dim=1)[0, facenet_id].cpu()))
                
                # Embedding 512D
                resnet.classify = False
                embs.append(resnet(t_img).cpu().numpy()[0])

            if len(embs) > 0 and len(confs) > 0:
                centroids[ident] = np.mean(embs, axis=0) # Punto medio della persona
                confidences[ident] = np.mean(confs)      # Sicurezza media

    valid_ids = list(centroids.keys())
    
    # ---------------------------------------------------------
    # LOGICA 1: THE LOOKALIKES (Sosia - Nearest Neighbors)
    # ---------------------------------------------------------
    # Scegliamo un Seed a caso (o il primo) e troviamo i 9 più vicini (Distanza Euclidea)
    seed_id = valid_ids[0]
    seed_emb = centroids[seed_id]
    distances = {i: np.linalg.norm(centroids[i] - seed_emb) for i in valid_ids}
    # Ordiniamo per distanza crescente e prendiamo i primi 10
    rr_lookalikes = sorted(distances, key=distances.get)[:10]

    # ---------------------------------------------------------
    # LOGICA 2: THE EXTREMES (5 Forti vs 5 Deboli)
    # ---------------------------------------------------------
    sorted_by_conf = sorted(confidences, key=confidences.get)
    rr_extremes = sorted_by_conf[:5] + sorted_by_conf[-5:] # 5 peggiori + 5 migliori

    # ---------------------------------------------------------
    # LOGICA 3: MAXIMUM DIVERSITY (K-Means)
    # ---------------------------------------------------------
    # Raggruppiamo lo spazio in 10 cluster distanti e prendiamo 1 persona per cluster
    X = np.array([centroids[i] for i in valid_ids])
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10).fit(X)
    rr_diversity = []
    for cluster_idx in range(10):
        # Troviamo l'ID del punto più vicino al centro del cluster
        cluster_center = kmeans.cluster_centers_[cluster_idx]
        dists = [np.linalg.norm(centroids[i] - cluster_center) for i in valid_ids]
        closest_id = valid_ids[np.argmin(dists)]
        rr_diversity.append(closest_id)

    # Salviamo le liste nel JSON
    out_dict = {
        "rr_lookalikes": rr_lookalikes,
        "rr_extremes": rr_extremes,
        "rr_diversity": rr_diversity
    }
    with open(out_json, 'w') as f:
        json.dump(out_dict, f, indent=4)
    print(f"\n[OK] Subsets calcolati e salvati in {out_json}")

if __name__ == "__main__":
    main()