import sqlite3
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np

import torch
import open_clip
import faiss

DB_PATH = "meta.sqlite"
FAISS_PATH = "index.faiss"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"


@torch.no_grad()
def main(query: str, k: int = 12):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    model = model.to(device).eval()

    # Load index
    index = faiss.read_index(FAISS_PATH)

    # Embed query text
    tokens = tokenizer([query]).to(device)
    feat = model.encode_text(tokens).float()
    feat = feat / feat.norm(dim=-1, keepdim=True)
    q = feat.cpu().numpy().astype(np.float32)

    # Search
    scores, ids = index.search(q, k)
    ids = ids[0].tolist()
    scores = scores[0].tolist()

    # Look up paths in SQLite
    conn = sqlite3.connect(DB_PATH)
    # Build map of id -> path for the returned ids
    # (fetching all at once is faster)
    placeholders = ",".join(["?"] * len(ids))
    rows = conn.execute(
        f"SELECT faiss_id, path FROM photos WHERE faiss_id IN ({placeholders})",
        ids
    ).fetchall()
    conn.close()

    id_to_path = {fid: path for fid, path in rows}

    # Print in ranked order
    for rank, (fid, score) in enumerate(zip(ids, scores), start=1):
        if fid == -1:
            continue
        path = id_to_path.get(fid, "<missing in db>")
        print(f"{rank:02d}. score={score:.3f}  {path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python search_photos.py "your search query" [k]')
        raise SystemExit(1)

    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) >= 3 else 12
    main(query, k)
