"""Verify the matched control and generate paper numbers directly from evidence."""
import csv
import hashlib
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

reference=read('evidence/misul-results.json')['bf16']
root=Path('artifacts/misul-dense-bf16')
summary=read(root/'summary.json');manifest=read(root/'manifest.json')
evaluation=read('evidence/misul-final-dense.json')
protocol=read('evidence/misul-dense-protocol.json')
assert summary['complete'] and manifest['complete'] and evaluation['complete']
assert digest(root/'training-state.safetensors')==summary['training_state_sha256']==evaluation['training_state_sha256']
for source,sha in manifest['source_sha256'].items():assert digest(source)==sha,source
for source,sha in protocol['source_sha256'].items():assert digest(source)==sha,source
for source,sha in protocol['reference_source_sha256'].items():assert digest(source)==sha,source
assert abs(summary['parameters']/reference['summary']['parameters']-1)<=.01
for key in ('data_sha256','tokenizer_sha256','program_sha256','seed','task_cycle','monitor_indices'):
    assert manifest[key]==reference['manifest'][key],key
for key in ('steps','batch','program_batch','mode','optimizer'):
    assert manifest['recipe'][key]==reference['manifest']['recipe'][key],key
for key in ('test_program_sha256','text_test_manifest_sha256','tokenizer_sha256'):
    assert evaluation[key]==reference['evaluation'][key],key
curves={'misul':[],'dense':[]}
with Path('evidence/misul-training-curve.csv').open() as f:
    for row in csv.DictReader(f):
        if row['precision']=='bf16':curves['misul'].append({k:float(row[k]) for k in ('step','elapsed_seconds','monitor_loss')})
guards=sorted(Path('evidence').glob('misul-dense-segment*-guard.json'),key=lambda p:int(re.search(r'segment(\d+)',p.name)[1]))
elapsed=0.;resources=[]
for path in guards:
    guard=read(path);resources.append(guard)
    for line in path.with_name(path.name.replace('-guard.json','.log')).read_text().splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        if 'monitor_loss' in row:curves['dense'].append(dict(step=row['step'],elapsed_seconds=elapsed+row['session_elapsed'],monitor_loss=row['monitor_loss']))
    elapsed+=guard['elapsed_seconds']
benchmarks={name:read(f'evidence/misul-architecture-benchmark-{name}.json') for name in ('misul','dense')}
for name,run in [('misul',reference['summary']),('dense',summary)]:
    assert benchmarks[name]['checkpoint_sha256']==run['training_state_sha256']
    assert benchmarks[name]['checkpoint_unchanged']
quality_ratio=reference['evaluation']['text_test']['loss']/evaluation['text_test']['loss']
train_ratio=reference['accounting']['actual_guarded_training_seconds']/elapsed
crossings={name:{str(target):next((r for r in curve if r['monitor_loss']<=target),None)
                  for target in protocol['thresholds']['time_to_quality_targets']} for name,curve in curves.items()}
result=dict(protocol=protocol,reference=reference,dense=dict(manifest=manifest,summary=summary,evaluation=evaluation),
 dense_accounting=dict(actual_guarded_training_seconds=elapsed,guards=[str(p) for p in guards],
                       peak_physical_bytes=max(g['sampled_peak_phys_bytes'] for g in resources),
                       interrupted_seconds=sum(g['elapsed_seconds'] for g in resources if g['returncode']!=0 or g['stop_reason'])),
 comparison=dict(parameter_relative_difference=summary['parameters']/reference['summary']['parameters']-1,
                 misul_relative_test_loss_change=quality_ratio-1,misul_training_time_ratio=train_ratio,
                 scope='One seed, common recipe, parameter-matched BF16 backbone comparison with shared task interfaces; baseline trained after reference tests were known.'),
 crossings=crossings,benchmarks=benchmarks)
Path('evidence/misul-architecture-results.json').write_text(json.dumps(result,indent=2))
macros={
 'DenseValidation':f"{summary['final_validation_loss']:.4f}",
 'DenseTest':f"{evaluation['text_test']['loss']:.4f}",
 'DenseBitsPerByte':f"{evaluation['text_test']['bits_per_byte']:.4f}",
 'DenseStreamHeld':f"{100*evaluation['held_out']['streams']['answer_accuracy']:.2f}",
 'DenseStreamLong':f"{100*evaluation['longer_lengths']['streams']['answer_accuracy']:.2f}",
 'DenseFlowEight':f"{100*evaluation['held_out']['flow']['8']['exact_accuracy']:.2f}",
 'DenseMinutes':f'{elapsed/60:.2f}',
}
loss_direction='lower' if quality_ratio<1 else 'higher'
time_direction='shorter' if train_ratio<1 else 'longer'
loss_text=f'{100*abs(quality_ratio-1):.2f}\\% {loss_direction}'
time_text=f'{100*abs(train_ratio-1):.2f}\\% {time_direction}'
mlong=100*reference['evaluation']['longer_lengths']['streams']['answer_accuracy']
dlong=100*evaluation['longer_lengths']['streams']['answer_accuracy']
hit_m,hit_d=crossings['misul']['5.5'],crossings['dense']['5.5']
target_text=(f"It reaches the fixed 5.5-nat validation-monitor target in {hit_m['elapsed_seconds']:.2f} seconds versus {hit_d['elapsed_seconds']:.2f} seconds for the dense control." if hit_m and hit_d else '')
macros['MisulArchitectureAbstract']=f'The combined model has {loss_text} official test loss at the same training budget. {target_text} Longer-program stream accuracy is {mlong:.2f}\\% versus {dlong:.2f}\\% for the dense control.'
macros['MisulArchitectureVerdict']=f'At this fixed budget, the combined model has {loss_text} official language-test loss. Total training execution is {time_text}; this cost includes the disclosed interrupted work. On longer programs, its stream answer accuracy is {mlong:.2f}\\% versus {dlong:.2f}\\%. These are observed differences between the two trained models, not evidence that every architectural component improves performance. Additional seeds, component ablations and independently tuned dense controls remain necessary.'
Path('paper/misul_architecture_macros.tex').write_text('% Generated from verified architecture evidence.\n'+'\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in macros.items())+'\n')
rows=[]
for target in ('6.0','5.5','5.0'):
    values=[]
    for name in ('misul','dense'):
        hit=crossings[name][target];values.append(f"{hit['elapsed_seconds']:.2f}" if hit else 'Not reached')
    rows.append(f'{target} nats/token & '+ ' & '.join(values)+r'\\')
timing=r'''\begin{table}[H]
\centering
\begin{tabularx}{\textwidth}{@{}Xrr@{}}
\toprule
First observed monitor target, seconds & \ModelName{} BF16 & Dense BF16\\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabularx}
\caption{Time to a fixed monitor quality, sampled every 512 updates. Each time
includes consumed training execution before the observed crossing.}
\end{table}
'''
rows=[]
for key,label,factor,unit in [('text_forward_context128_batch1','128-token forward, batch 1',1000,'ms'),('uncached_greedy32_tokens_batch1','Generate 32 tokens, batch 1',1,'s'),('flow_length6_batch8_passes8','Eight-pass Flow, batch 8',1000,'ms')]:
    values=[f"{benchmarks[name]['cases'][key]['median_seconds']*factor:.2f}" for name in ('misul','dense')]
    rows.append(f'{label} ({unit}) & '+' & '.join(values)+r'\\')
timing+=r'''\begin{table}[H]
\centering
\begin{tabularx}{\textwidth}{@{}Xrr@{}}
\toprule
Inference latency, median & \ModelName{} BF16 & Dense BF16\\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule
\end{tabularx}
\caption{Eighteen completed-device repetitions per workload across two
counterbalanced processes per model, after separately recorded first executions. Both models use BF16 and the same uncached generation
procedure. These are local implementation measurements.}
\end{table}
'''
Path('paper/misul_architecture_timing.tex').write_text(timing)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#687386','text.color':'#171E2B','axes.labelcolor':'#171E2B','grid.color':'#D7DCE2','grid.linewidth':.5})
fig,axes=plt.subplots(1,2,figsize=(7.05,2.55),layout='constrained')
for name,label,color,style in [('misul','Interlace','#0866C6','-'),('dense','Dense control','#687386','--')]:
    curve=curves[name]
    axes[0].plot([r['step']*256/1e6 for r in curve],[r['monitor_loss'] for r in curve],label=label,color=color,linestyle=style,linewidth=1.5)
    axes[1].plot([r['elapsed_seconds']/60 for r in curve],[r['monitor_loss'] for r in curve],label=label,color=color,linestyle=style,linewidth=1.5)
axes[0].set(xlabel='Retained language target tokens (millions)',ylabel='Monitor loss (nats/token)')
axes[1].set(xlabel='Consumed training execution (minutes)',ylabel='Monitor loss (nats/token)')
for ax in axes:ax.grid(axis='y');ax.legend(frameon=False,fontsize=8)
fig.savefig('paper/figures/misul_architecture.pdf');fig.savefig('paper/figures/misul_architecture.png',dpi=180)
print(json.dumps(result['comparison'],indent=2))
