import regex as re 
import json
#import heapq
import os
import glob
from multiprocessing import Pool 

GPT2_REGEX = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")

class Tokenizer:

    def __init__(self):
        self.merge_rules = {}    # for encode
        self.vocab = self._build_vocab() # for decode : bytes - string 
    

    def _build_vocab(self):
        vocab ={}
        for i in range(256):
            vocab[i] = bytes([i])
        return vocab
    

    def merge(self, ids, pair, idx):  # iterate through some ids, if pair in it, replace it with idx
        new_ids = [] 
        i = 0
        while i < len(ids):
            if i<len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i+=2
            else:
                new_ids.append(ids[i])
                i+=1
        return tuple(new_ids) # tuple as key of dict 
    

    def train(self, data_dir, vocab_size):

        word_freq = {}   # word frequency counter
        line_cnt = 0
        txt_files = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        for file_idx, path in enumerate(txt_files):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:    #interator ; memory is cleared after done
                    line_cnt += 1
                    tokens = GPT2_REGEX.findall(line)
                    tokens = [t for t in tokens if t] # null char
                    for token in tokens:
                        byte_tuple = tuple(token.encode('utf-8'))
                        word_freq[byte_tuple] = word_freq.get(byte_tuple, 0) + 1  #'hello world' : {(104, 101, 108, 108, 111): 1, (32, 119, 111, 114, 108, 100): 1}
            if file_idx % 100 ==0:
                print(f'finished : {file_idx} files')


        pair_freq = {}  #token pair frequencey counter
        pair_to_words = {}  # looking for token pair in all words is very expensive; we create this dict as an index to know what words to look at for particular tokens 
    

        for byte_tuple, freq in word_freq.items():
            for i in range (len(byte_tuple) -1):
                pair = (byte_tuple[i], byte_tuple[i+1])
                pair_freq[pair] = pair_freq.get(pair, 0) + freq   # ‘the cat the dog' : {(116, 104): 2, (104, 101): 2, (32, 99): 1, (99, 97): 1, (97, 116): 1, (32, 116): 1, (32, 100): 1, (100, 111): 1, (111, 103): 1}

                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(byte_tuple)

        #heap = [(-freq, pair) for pair, freq in pair_freq.items()] #built-in heap only supports min-heap 
        #heapq.heapify(heap)

        for i in range(vocab_size - 256): # number of merges
            print(f'merging step {i}')
            if not pair_freq: # no pairs to merge
                break

            # while heap:
            #     freq, best_pair = heapq.heappop(heap)
            #     actual_freq = pair_freq.get(best_pair, 0) 
            #     if actual_freq == -freq:  #lazy deletion ： checking if the best pair is valid (pair might have been affected)
            #         break
            # else:
            #     break    # the heap is empty
        
            new_id = i + 256
            best_pair = max(pair_freq, key=pair_freq.get)

            self.merge_rules[best_pair] = new_id 
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # find affected words from affected tokens
            affected_words = pair_to_words.pop(best_pair, set())
            if not affected_words:
                continue

            #updating pair_freq, but only for affected words
            #removing the contribution of old words
            for w in affected_words:
                f = word_freq[w]  # frequencey of the word
                for p in range(len(w)-1):
                    pair = (w[p], w[p+1])
                    pair_freq[pair] -= f #upadtaing token pair frequency 
                    if pair_freq[pair]==0:
                        del pair_freq[pair]
                    
                    if pair in pair_to_words:
                        pair_to_words[pair].discard(w)  # word not exists
                        if not  pair_to_words[pair]:
                            del pair_to_words[pair]

            #updating word_freq
            #adding the contribution of new words
            new_word_freq = {}
            for w in affected_words:
                f = word_freq[w]
                new_word = self.merge(w,best_pair,new_id)
                new_word_freq[new_word] = new_word_freq.get(new_word,0) + f
                for j in range(len(new_word)-1):
                    pair = (new_word[j], new_word[j+1])
                    pair_freq[pair] = pair_freq.get(pair, 0) + f

                    if pair not in pair_to_words:
                        pair_to_words[pair] = set()
                    pair_to_words[pair].add(new_word)
                    #heapq.heappush(heap, (-pair_freq[pair], pair))

            #removing old words, adding new words
            for word in affected_words:
                del word_freq[word]

            for new_word, freq in new_word_freq.items():
                word_freq[new_word] = word_freq.get(new_word,0) + freq
        

    def encode(self,text):
        tokens = GPT2_REGEX.findall(text)
        ids = []

        for token in tokens:
            word_ids = list(token.encode('utf-8'))

            while (len(word_ids) >= 2):
                best_pair = None
                best_rank = float('inf')

                for i in range(len(word_ids)-1):
                    pair = (word_ids[i], word_ids[i+1])
                    if pair in self.merge_rules:
                        rank = self.merge_rules[pair]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair

                if best_pair is None:
                    break
                #best_pair = min(pairs_to_merge, key=lambda x: self.merge_rules[x]) #smallest idx (same order as the training)

                new_word_ids = []
                i =0 
                while i<len(word_ids):
                    if i < (len(word_ids)-1) and word_ids[i] == best_pair[0] and word_ids[i+1] == best_pair[1]:
                        new_word_ids.append(self.merge_rules[best_pair])  # 
                        i += 2
                    else:
                        new_word_ids.append(word_ids[i])
                        i += 1
                word_ids = new_word_ids
            ids.extend(word_ids)

        return ids 
    

    def decode(self, ids):
        text_bytes = b''.join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode('utf-8', errors='replace')
        return text
    

    def save(self, path):
        #json only take string as key, while self.merge_rules takes tuple 
        # 1st step: convert tuple key into string
        save_rules = {}
        for pair, idx in self.merge_rules.items():
            #print(pair)
            key = f'{pair[0]},{pair[1]}'
            save_rules[key] = idx
        
        #self.vocab takes bytes as value (b'th'), convert them into list
        save_vocab = {}
        for idx, text_bytes in self.vocab.items():
            save_vocab[str(idx)] = list(text_bytes)

        data_to_save = {
            'merge_rules' : save_rules,
            'vocab' : save_vocab
        }
        with open(path,'w', encoding='utf-8') as f:
            json.dump(data_to_save, f , indent=2)
        print(f'tokenizer saved to {path}')

    
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        # convert string back to tuples
        self.merge_rules = {}
        for key, val in loaded_data['merge_rules'].items():
            pair = tuple(int(x) for x in key.split(','))
            self.merge_rules[pair] = val
        
        # convert list back to bytes
        self.vocab ={}
        for key, val in loaded_data['vocab'].items():
            key = int(key)
            val = bytes(val)
            self.vocab[key] = val
        
        print(f'tokenizer loaded from {path}')


#-------------------------------------------------------------------------------------------------------------------
#tokenizer = Tokenizer()
# tokenizer.train(r'C:\Users\xhu70\Documents\LLM_from_scratch\data', vocab_size=50304)
# tokenizer.save('my_tokenizer.json')
# tokenizer.encode('hellow world')
# print(tokenizer.decode(tokenizer.encode('hellow world')) == 'hellow world')
          
