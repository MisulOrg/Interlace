import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np


class KnowledgeData:
    templates = {
        "train": ("what is {r} of {s} ?", "for {s} what is {r} ?", "{s} has {r} equal to", "the {r} for {s} is"),
        "validation": ("{r} for {s} is what ?",),
        "test": ("for {s} the {r} is what ?",),
    }

    def __init__(self, subjects=256, relations=32, objects=256, seed=11, kind="random"):
        self.subjects, self.relations, self.objects = subjects, relations, objects
        rng = np.random.default_rng(seed)
        if kind == "random":
            self.facts = rng.integers(objects, size=(subjects, relations))
        elif kind == "structured":
            subject_codes = rng.permutation(subjects) % objects
            offsets = rng.integers(objects, size=relations)
            object_codes = rng.permutation(objects)
            self.facts = object_codes[(subject_codes[:, None] + offsets[None, :]) % objects]
        else:
            raise ValueError(kind)
        words = set()
        for template in self.templates["train"]:
            words.update(template.format(s="", r="").split())
        words.update(f"s{i}" for i in range(subjects))
        words.update(f"r{i}" for i in range(relations))
        words.update(f"o{i}" for i in range(objects))
        self.vocab = {word: i for i, word in enumerate(["<pad>", "<unk>", "<bos>"] + sorted(words))}
        self.digest = hashlib.sha256(self.facts.tobytes()).hexdigest()

    def examples(self, split="train", max_length=12):
        templates=self.templates[split]
        count=self.subjects*self.relations
        identities=np.indices((self.subjects,self.relations),dtype=np.int32).reshape(2,-1).T
        subject_tokens=np.array([self.vocab[f's{i}'] for i in range(self.subjects)],np.int32)
        relation_tokens=np.array([self.vocab[f'r{i}'] for i in range(self.relations)],np.int32)
        object_tokens=np.array([self.vocab[f'o{i}'] for i in range(self.objects)],np.int32)
        prompts=np.zeros((count*len(templates),max_length),dtype=np.int32)
        for t,template in enumerate(templates):
            words=['<bos>']+template.format(s='__SUBJECT__',r='__RELATION__').split()
            if len(words)>max_length:raise ValueError('prompt exceeds context')
            for j,word in enumerate(words,start=max_length-len(words)):
                if word=='__SUBJECT__':value=subject_tokens[identities[:,0]]
                elif word=='__RELATION__':value=relation_tokens[identities[:,1]]
                else:value=self.vocab.get(word,1)
                prompts[t::len(templates),j]=value
        answers=np.repeat(object_tokens[self.facts.reshape(-1)],len(templates))
        ids=np.repeat(identities,len(templates),axis=0)
        return prompts,answers,ids


class NamedKnowledgeData(KnowledgeData):
    """A frozen, single-relation named snapshot using the same recall templates."""
    def __init__(self,path):
        raw=Path(path).read_bytes();payload=json.loads(raw);records=payload['records']
        if not records:raise ValueError('empty named fact dataset')
        object_ids=sorted({row['country'] for row in records})
        index={value:i for i,value in enumerate(object_ids)}
        super().__init__(len(records),1,len(object_ids),seed=0)
        self.facts=np.array([[index[row['country']]] for row in records],dtype=np.int64)
        self.digest=hashlib.sha256(self.facts.tobytes()).hexdigest()
        normalize=lambda value:unicodedata.normalize('NFC',value).casefold().strip()
        self.input_aliases={normalize(row['cityLabel']):f's{i}' for i,row in enumerate(records)}
        if len(self.input_aliases)!=len(records):raise ValueError('ambiguous named subjects')
        relation=normalize(payload['relation'])
        if relation in self.input_aliases:raise ValueError('relation aliases a subject')
        self.input_aliases[relation]='r0'
        names={row['country']:row['countryLabel'] for row in records}
        self.output_labels=[names[value] for value in object_ids]
        self.fact_source={key:payload[key] for key in ['source','license','retrieved_utc','raw_sha256']}
        self.fact_source['prepared_sha256']=hashlib.sha256(raw).hexdigest()


def add_distractor_prefix(prompts,identities,subject_tokens,relation_tokens,rng,*,bos=2,question,probability=.5):
    """Add an earlier three-token query inside BOS; keep the main query intact."""
    result=prompts.copy()
    rows=np.flatnonzero(rng.random(len(prompts))<probability)
    if not len(rows):return result
    markers=prompts[rows]==bos
    if np.any(markers.sum(axis=1)!=1):raise ValueError('one BOS marker required')
    columns=markers.argmax(axis=1)
    if np.any(columns<3):raise ValueError('not enough prefix space')
    ds=rng.integers(1,len(subject_tokens),size=len(rows)) if len(subject_tokens)>1 else np.zeros(len(rows),dtype=int)
    dr=rng.integers(1,len(relation_tokens),size=len(rows)) if len(relation_tokens)>1 else np.zeros(len(rows),dtype=int)
    result[rows,columns-3]=bos
    result[rows,columns-2]=subject_tokens[(identities[rows,0]+ds)%len(subject_tokens)]
    result[rows,columns-1]=relation_tokens[(identities[rows,1]+dr)%len(relation_tokens)]
    result[rows,columns]=question
    return result
