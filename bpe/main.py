import multiprocessing as mp
import numpy as np
import os
import glob
from base import Tokenizer

def worker_encode_init(tokenizer_path): # 
    # in python, data are transimitted among thredas with pickle
    # pickle converts python obejects (dict in this case) into binary to share and transmit
    # dict is too large
    # so we initiailizer the tokenzier at each thread; call it a global variable
    global worker_tokenizer
    worker_tokenizer = Tokenizer()
    worker_tokenizer.load(tokenizer_path)

def process_file(filepath):
    global worker_tokenizer
    ids = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line_ids = worker_tokenizer.encode(line)
            ids.extend(line_ids)

            # if (idx+1) %100 ==0:
            #     print(f" processed {idx+1} lines", flush=True)s

    return ids

if __name__ == "__main__":
    data_dir = r'C:\Users\xhu70\Documents\LLM_from_scratch\data'
    tokenizer_path = 'my_tokenizer.json'
    output = 'books_data.bin' 

    texts = glob.glob(os.path.join(data_dir, "*.txt"))
    print(f'{len(texts)} files to tokenize')
    memmap = np.memmap(output, dtype=np.uint16, mode='w+',shape=(50000000000,))

    num_workers = 8
    offset = 0
    with mp.Pool(processes=num_workers, initializer=worker_encode_init, initargs=(tokenizer_path,)) as pool:
        for i, chunk in enumerate(pool.imap_unordered(process_file, texts, chunksize=5)):
            n = len(chunk)
            memmap[offset:offset + n] = chunk   
            memmap.flush()                      
            offset += n
            print(f">>> [{i+1}] wrote {n:,} tokens, total so far: {offset:,}", flush=True)
    actual = np.memmap(output, dtype=np.uint16, mode='r+', shape=(offset,))

    final = np.memmap('books_data_final.bin', dtype=np.uint16, mode='w+', shape=(offset,))
    final[:] = actual
    final.flush()
    print(f"Total tokens: {len(final):,}")
