"""Run text, synchronous streams, or visible Flow from one packed checkpoint."""
import argparse
import json
from pathlib import Path
import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer
from .misul import MisulModel
from .misul_flow import sample_flow,stability_score
from .safe_run import require_guard
from .stream_program import rollout


def main():
    require_guard()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--mode',choices=['text','streams','flow'],required=True)
    parser.add_argument('--prompt',default='The city of')
    parser.add_argument('--tokens',type=int,default=64)
    parser.add_argument('--temperature',type=float,default=.7)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--digits',default='2,5,3,7')
    parser.add_argument('--flow-steps',type=int,default=8)
    args=parser.parse_args()
    if not 1<=args.tokens<=256 or not 0<=args.temperature<=2 or not 1<=args.flow_steps<=32:
        parser.error('tokens1-256, temperature0-2, and Flow steps1-32 are required')
    mx.set_memory_limit(1000*2**20);mx.set_cache_limit(32*2**20)
    directory=Path(args.model);model=MisulModel.load(directory)
    tokenizer=Tokenizer.from_file(str(directory/'tokenizer.json'))
    if args.mode=='text':
        token_ids=tokenizer.encode(args.prompt).ids
        if not token_ids:parser.error('provide a nonempty prompt')
        print(args.prompt,end='',flush=True);continuation=[];printed=''
        for index in range(args.tokens):
            x=mx.array([token_ids[-model.config['context']:]],dtype=mx.int32)
            logits=model(x)[0,-1].astype(mx.float16)
            # Stream control tokens are not natural-language output symbols.
            for name in ('<|idle|>','<|end-input|>'):
                control=tokenizer.token_to_id(name)
                if control is not None:logits[control]=-float('inf')
            token=int(mx.argmax(logits)) if args.temperature==0 else int(mx.random.categorical(logits/args.temperature,key=mx.random.key(args.seed+index)))
            token_ids.append(token);continuation.append(token)
            text=tokenizer.decode(continuation)
            if text.endswith('\ufffd'):continue
            if text.startswith(printed):print(text[len(printed):],end='',flush=True);printed=text
        print();return
    try:digits=[int(d.strip()) for d in args.digits.split(',')]
    except ValueError:parser.error('provide comma-separated digits0-9')
    if not 1<=len(digits)<=min(16,model.config['context']-3) or any(not 0<=d<=9 for d in digits):
        parser.error('provide1-16 digits0-9 within the model context')
    digit_ids=[tokenizer.encode(str(i)).ids[0] for i in range(10)]
    idle=tokenizer.token_to_id('<|idle|>');end=tokenizer.token_to_id('<|end-input|>')
    if args.mode=='streams':
        external=np.full((1,len(digits)+3),idle,dtype=np.int32)
        external[0,1:len(digits)+1]=[digit_ids[d] for d in digits]
        external[0,len(digits)+1]=end
        predicted=rollout(model,external,idle)[0]
        for tick,row in enumerate(predicted):
            print(json.dumps(dict(tick=tick,**{name:tokenizer.decode([int(value)],skip_special_tokens=False)
                for name,value in zip(('input','thought','output'),row)})),flush=True)
    else:
        clues=mx.array([[digit_ids[d] for d in digits]],dtype=mx.int32)
        candidate,_,trace=sample_flow(model,clues,args.flow_steps,args.seed)
        for row in trace:print(f"pass {row['step']:2d}  t={row['time']:.3f}  "+' '.join(map(str,row['candidate'][0])),flush=True)
        score=stability_score(model,clues,candidate,mx.ones(clues.shape,dtype=mx.bool_),seed=args.seed+1)
        print('Predicted prefix states: '+' '.join(map(str,candidate.tolist()[0])))
        print(f'Re-noise consistency loss: {float(score.item()):.4f} (consistency alone does not establish correctness)')


if __name__=='__main__':main()
