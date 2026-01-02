import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sqlite3
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import open_clip
import faiss

# ------------ Config ------------
DB_PATH = "meta.sqlite"
FAISS_PATH = "index.faiss"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}  # start simple
BATCH_SIZE = 32
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
EMBED_DIM = 512  # ViT-B-32 produces 512-d embeddings


# ------------ Helpers ------------
def iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS photos (
            faiss_id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            mtime REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_path ON photos(path)")
    conn.commit()


def load_faiss_index() -> faiss.Index:
    """
    Use IndexIDMap2 so we can add vectors with explicit IDs and remove/update them.
    """
    if os.path.exists(FAISS_PATH):
        return faiss.read_index(FAISS_PATH)

    base = faiss.IndexFlatIP(EMBED_DIM)  # inner product on normalized vectors = cosine similarity
    index = faiss.IndexIDMap2(base)
    return index


@torch.no_grad()
def embed_images(
    model,
    preprocess,
    device: str,
    image_paths: List[str],
) -> np.ndarray:
    """
    Returns NxD float32 normalized image embeddings.
    """
    tensors = []
    for sp in image_paths:
        img = Image.open(sp).convert("RGB")
        tensors.append(preprocess(img))

    batch = torch.stack(tensors).to(device)
    feats = model.encode_image(batch).float()
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


def main(root_dir: str):
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Folder does not exist: {root}")

    # Device + Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()

    # DB
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    # FAISS
    index = load_faiss_index()
    #FAISS -> stores all the image embeddings for searching later

    # 1) Remove missing files (present in DB but no longer on disk)
    cur = conn.execute("SELECT faiss_id, path FROM photos")
    rows = cur.fetchall()

    missing_ids = []
    for faiss_id, path in rows:
        if not os.path.exists(path): #if the image path does exist in the filesystem
            missing_ids.append(faiss_id)

    if missing_ids:
        ids_np = np.array(missing_ids, dtype=np.int64) #must convert for FAISS function (remove_ids) to work
        index.remove_ids(ids_np)
        conn.executemany("DELETE FROM photos WHERE faiss_id = ?", [(i,) for i in missing_ids]) #remove from DB too
        conn.commit()
        print(f"Removed {len(missing_ids)} missing files from index.")

    # 2) Find new or changed files to (re)index
    # We'll build two lists:
    # - new_paths: paths not in DB (insert -> get new faiss_id)
    # - changed: (faiss_id, path) where mtime changed (update same faiss_id)
    db_map = {path: (faiss_id, mtime) for faiss_id, path, mtime in conn.execute( 
        "SELECT faiss_id, path, mtime FROM photos"
    ).fetchall()}
    
    #db_map with the photo path as key and (faiss_id, mtime) as value

    new_paths: List[str] = []
    changed: List[Tuple[int, str, float]] = []

    for p in iter_images(root):
        sp = str(p)
        mtime = p.stat().st_mtime
        if sp not in db_map:
            new_paths.append(sp)
        else:
            faiss_id, old_mtime = db_map[sp]
            if float(old_mtime) != float(mtime):
                changed.append((faiss_id, sp, mtime))

    print(f"Found {len(new_paths)} new images, {len(changed)} changed images.")

    # 3) Insert new files into DB first to get stable faiss_id
    new_items: List[Tuple[int, str, float]] = []
    if new_paths:
        for sp in new_paths:
            mtime = os.path.getmtime(sp)
            cur = conn.execute("INSERT OR IGNORE INTO photos(path, mtime) VALUES(?, ?)", (sp, mtime))
            conn.commit()
            # If it already existed (race), fetch id
            if cur.lastrowid:
                faiss_id = cur.lastrowid
            else:
                faiss_id = conn.execute("SELECT faiss_id FROM photos WHERE path = ?", (sp,)).fetchone()[0]
            new_items.append((faiss_id, sp, mtime))

    # 4) For changed files: update DB mtime now (we'll update vector in FAISS too)
    if changed:
        conn.executemany("UPDATE photos SET mtime = ? WHERE faiss_id = ?", [(mt, fid) for fid, _, mt in changed])
        conn.commit()

    # Combine items to embed (new + changed)
    to_embed: List[Tuple[int, str]] = [(fid, sp) for fid, sp, _ in new_items] + [(fid, sp) for fid, sp, _ in changed]
    if not to_embed:
        print("Nothing to index. You're up to date.")
        faiss.write_index(index, FAISS_PATH)
        conn.close()
        return

    # 5) Embed in batches and update FAISS
    print(f"Embedding {len(to_embed)} images on {device}...")
    for i in tqdm(range(0, len(to_embed), BATCH_SIZE)):
        batch = to_embed[i:i + BATCH_SIZE]
        ids = np.array([fid for fid, _ in batch], dtype=np.int64)
        paths = [sp for _, sp in batch]

        # Compute embeddings (may throw if an image is corrupt)
        good_ids = []
        good_paths = []
        for fid, sp in batch:
            try:
                # quick verify image loads
                Image.open(sp).verify()
                good_ids.append(fid)
                good_paths.append(sp)
            except Exception:
                # Remove from DB if new file was bad
                conn.execute("DELETE FROM photos WHERE faiss_id = ?", (fid,))
                conn.commit()

        if not good_paths:
            continue

        feats = embed_images(model, preprocess, device, good_paths)

        # If these were updates, remove old vectors first
        index.remove_ids(np.array(good_ids, dtype=np.int64))
        index.add_with_ids(feats, np.array(good_ids, dtype=np.int64))

    # Save everything
    faiss.write_index(index, FAISS_PATH)
    conn.close()
    print(f"Done. Saved {FAISS_PATH} and {DB_PATH}.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python index_photos.py /path/to/photo/folder")
        raise SystemExit(1)
    main(sys.argv[1])
