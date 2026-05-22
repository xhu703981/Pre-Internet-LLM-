import os
import regex as re
from collections import Counter

##To be able to train on a large datasets: I implemented
##1. frequency counter: instead of dealing with the full length text, we only care about the words appeard, which, in this case, were stored as tuples
##2. doubly linked list: to be able to perform merge, we treat each word as a dll, which also prevents cross-words tokenization 

GPT2_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class Node:
    def __init__(self,val,freq):
        self.next=None
        self.prev=None
        self.val=val
        self.merged=False
        self.freq = freq 

def ids_to_dll(ids,freq=1):
    head=Node(ids[0],freq)
    current_node=head
    for i in range(1, len(ids)):
        new_node=Node(ids[i],freq)
        current_node.next=new_node
        new_node.prev=current_node
        current_node=new_node
    return head

def build_pair_data(head):
    pair_counts={}
    pair_position={}
    node=head
    while node.next:
        pair=(node.val,node.next.val)
        pair_counts[pair]=pair_counts.get(pair,0)+1
        if pair not in pair_position:
            pair_position[pair]=set()
        pair_position[pair].add(node)
        node=node.next
    return pair_counts, pair_position

def merge(ids,pair,idx):
    i=0
    new_ids=[]
    while i< len(ids):
        if  i<len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
            i+=2
            new_ids.append(idx)
        else:
            new_ids.append(ids[i])
            i+=1
    return new_ids

def merge_node(pair,idx,pair_counts,pair_positions):
    for node in list(pair_positions[pair]):
        if node.merged or node.next is None or node.next.merged: continue
        if (node.val,node.next.val)!=pair: continue
        L=node.prev
        R=node.next.next
        if L:
            if (L.val,node.val) in pair_counts:
                pair_counts[(L.val,node.val)]-=node.freq
                pair_positions[(L.val, node.val)].discard(L)
                if pair_counts[(L.val, node.val)] <= 0:
                    del pair_counts[(L.val, node.val)]
                    del pair_positions[(L.val, node.val)]
        if R:
            if (node.next.val,R.val) in pair_counts:
                pair_counts[(node.next.val,R.val)]-=node.freq
                pair_positions[(node.next.val,R.val)].discard(node.next)
                if pair_counts[(node.next.val, R.val)] <= 0:
                    del pair_counts[(node.next.val, R.val)]
                    del pair_positions[(node.next.val, R.val)]
        if pair in pair_counts:
            pair_counts[pair]-=node.freq
            pair_positions[pair].discard(node)
            if pair_counts[pair] <= 0:
                del pair_counts[pair]
                del pair_positions[pair]
        old_next = node.next
        node.val=idx
        node.next=R
        old_next.merged=True
        if node.next:
            node.next.prev=node
        if L:
            new_pair1=(L.val, idx)
            if new_pair1 not in pair_positions: pair_positions[new_pair1]=set()
            pair_counts[new_pair1] = pair_counts.get(new_pair1, 0) + node.freq
            pair_positions[new_pair1].add(L)

        if R:
            new_pair2=(idx,R.val)
            if new_pair2 not in pair_positions: pair_positions[new_pair2]=set()
            pair_counts[new_pair2] = pair_counts.get(new_pair2, 0) + node.freq
            pair_positions[new_pair2].add(node)

class BPETokenizer:
    def __init__(self):
        self.vocab={}
        self.merge_rules={}
    
    def train(self,data_dir,vocab_size):
        word_freq=Counter()
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                    for token in re.findall(GPT2_PATTERN,f.read()):
                        word_freq[tuple(token.encode("utf-8"))]+=1

        word_heads = {}
        pair_counts = {}
        pair_positions = {}
        for word, freq in word_freq.items():
            head = ids_to_dll(list(word),freq)
            word_heads[word] = (head, freq)
            node = head
            while node.next:
                pair = (node.val, node.next.val)
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                if pair not in pair_positions:
                    pair_positions[pair] = set()
                pair_positions[pair].add(node)
                node = node.next

        num_merges=vocab_size-256
        for i in range(num_merges):
            pair=max((p for p in pair_counts if pair_counts[p] > 0), key=lambda p: pair_counts[p])
            merge_node(pair,256+i,pair_counts,pair_positions)
            self.merge_rules[pair]=256+i
            print(f"merge {i}: {pair} -> {256+i}")
        for i in range(256):
            self.vocab[i]=bytes([i])
        for pair, idx in self.merge_rules.items():
            self.vocab[idx]=self.vocab[pair[0]]+self.vocab[pair[1]]
    
    def encode(self,text):
        ids=[]
        for token in re.findall(GPT2_PATTERN,text):
            word_ids=list(token.encode("utf-8"))
            while True:
                pairs = [(word_ids[i], word_ids[i+1]) for i in range(len(word_ids)-1)]
                best = min(
                (p for p in pairs if p in self.merge_rules),
                key=lambda p: self.merge_rules[p],
                default=None
                )   
                if best is None: break
                else: word_ids=merge( word_ids,best, self.merge_rules[best])
            ids.extend(word_ids)
        return ids
    
    def decode(self,ids):
        text=[]
        for id in ids:
            text.append(self.vocab[id])
        return b"".join(text).decode("utf-8")
    
    def save(self, path):
      import json
      data = {
          "merge_rules": {f"{p[0]},{p[1]}": idx for p, idx in self.merge_rules.items()}
      }
      with open(path, "w", encoding="utf-8") as f:
          json.dump(data, f)

    def load(self, path):
      import json
      with open(path, "r", encoding="utf-8") as f:
          data = json.load(f)
      self.merge_rules = {
          tuple(int(x) for x in k.split(",")): v
          for k, v in data["merge_rules"].items()
      }
      self.vocab = {}
      for i in range(256):
          self.vocab[i] = bytes([i])
      for pair, idx in sorted(self.merge_rules.items(), key=lambda x: x[1]):
          self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

# if __name__ == "__main__":
#       tokenizer = BPETokenizer()
#       tokenizer.train("the quick brown fox jumps over the lazy dog. " * 200, vocab_size=276)
#       encoded = tokenizer.encode("the quick brown fox")
#       decoded = tokenizer.decode(encoded)
#       print(decoded)
#       print(decoded == "the quick brown fox")
#       print("Vocab size:", len(tokenizer.vocab))