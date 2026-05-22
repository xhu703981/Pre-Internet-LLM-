from base import BPETokenizer
import os
import numpy as np
from multiprocessing import Pool
import functools

def encode_file(args):
    filepath, tokenizer_path = args
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)  # 每个进程独立加载
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return tokenizer.encode(text)

if __name__ == "__main__":
    data_dir = r"C:\Users\xhu70\Documents\LLM_from_scratch\data"
    tokenizer_path = "tokenizer.json"
    print(f"cpu count{os.cpu_count()}")

    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
    args = [(f, tokenizer_path) for f in files]

    out = open("data.bin", "wb")
    total_tokens = 0

    with Pool(processes=12) as pool:
        for i, ids in enumerate(pool.imap(encode_file, args, chunksize=10)):
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(out)
            total_tokens += len(ids)
            if i % 100 == 0:
                print(f"{i}/{len(files)} done, tokens: {total_tokens}")

    out.close()
    print(f"Total tokens: {total_tokens}")