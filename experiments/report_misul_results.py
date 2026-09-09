"""Build the local results record and scientific figure from completed runs."""
import csv
import hashlib
import json
from pathlib import Path
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def collect(label):
    root = Path(f'artifacts/misul-joint-checked-{label}' + ('-recovered' if label == 'bf16' else ''))
    summary, manifest = [read(root / name) for name in ('summary.json', 'manifest.json')]
    evaluation = read(f'evidence/misul-final-{label}.json')
    assert summary['complete'] and manifest['complete'] and evaluation['complete']
    assert summary['steps'] == 16384
    assert evaluation['training_state_sha256'] == summary['training_state_sha256'] == digest(root / 'training-state.safetensors')
    guards = list(Path('evidence').glob(f'misul-joint-checked-{label}-segment*-guard.json'))
    if label == 'bf16':
        guards += list(Path('evidence').glob('misul-joint-checked-bf16-recovered-segment*-guard.json'))
    guards.sort(key=lambda p: ('recovered' in p.name, int(re.search(r'segment(\d+)', p.name)[1])))
    elapsed = 0.
    curve = []
    resource = []
    for path in guards:
        guard = read(path)
        resource.append(guard)
        for line in path.with_name(path.name.replace('-guard.json', '.log')).read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'monitor_loss' in row:
                curve.append(dict(precision=label, step=row['step'],
                                  elapsed_seconds=elapsed + row['session_elapsed'],
                                  monitor_loss=row['monitor_loss']))
        elapsed += guard['elapsed_seconds']
    overflow = root / 'overflow.jsonl'
    accounting = dict(actual_guarded_training_seconds=elapsed,
                      interrupted_seconds=sum(g['elapsed_seconds'] for g in resource
                                              if g['stop_reason'] is not None or g['returncode'] != 0),
                      peak_physical_bytes=max(g['sampled_peak_phys_bytes'] for g in resource),
                      largest_reported_swap_endpoint_change_bytes=max(g['last_sampled_swap_bytes'] - g['initial_swap_bytes'] for g in resource),
                      swap_growth_guard_stops=sum(g['stop_reason'] == 'system swap growth' for g in resource),
                      swap_measurement='Guard reports initial and final sampled system swap; these endpoint differences do not reconstruct the maximum during each run.',
                      observed_overflow_attempts=len(overflow.read_text().splitlines()) if overflow.exists() else 0,
                      evaluation_seconds=evaluation['elapsed_seconds'],
                      guards=[str(p) for p in guards])
    return dict(summary=summary, manifest=manifest, evaluation=evaluation, accounting=accounting,
                root=str(root)), curve


fp8, fp8_curve = collect('fp8')
bf16, bf16_curve = collect('bf16')
for key in ('source_sha256', 'data_sha256', 'tokenizer_sha256', 'program_sha256', 'seed', 'parameters'):
    assert fp8['manifest'][key] == bf16['manifest'][key], key
for key in ('configuration',):
    configs = [{k: v for k, v in r['manifest'][key].items() if k != 'precision'} for r in (fp8, bf16)]
    assert configs[0] == configs[1]
for key in ('steps', 'batch', 'program_batch', 'mode', 'optimizer'):
    assert fp8['manifest']['recipe'][key] == bf16['manifest']['recipe'][key], key
assert fp8['evaluation']['test_program_sha256'] == bf16['evaluation']['test_program_sha256']
assert fp8['evaluation']['text_test_manifest_sha256'] == bf16['evaluation']['text_test_manifest_sha256']
release = Path('artifacts/misul-joint-checked-fp8/release')
storage = read(release / 'storage.json')
assert storage['weight_sha256'] == digest(release / 'weights.safetensors')
assert fp8['summary']['export']['reload_exact']
loss_change = fp8['summary']['final_validation_loss'] / bf16['summary']['final_validation_loss'] - 1
result = dict(fp8=fp8, bf16=bf16,
              precision_gate=dict(relative_validation_loss_change=loss_change,
                                  threshold=.01, passed=loss_change <= .01,
                                  relative_test_loss_change=fp8['evaluation']['text_test']['loss'] / bf16['evaluation']['text_test']['loss'] - 1,
                                  scope='One local seed with the explicitly amended BF16 fractional-scale recovery; fixed architecture/data/budget. The original scale-floor-1 control failed. No universal quality guarantee.'),
              numerical_amendment=read('evidence/misul-fractional-amendment.json'),
              numerical_recovery_completion=read('evidence/misul-fractional-completion.json'),
              storage=storage | dict(complete_release_bytes=sum(p.stat().st_size for p in release.iterdir() if p.is_file()),
                                     equal_parameter_bf16_bytes=2 * storage['parameters'],
                                     tensor_reduction_vs_bf16=1 - storage['tensor_bytes'] / (2 * storage['parameters'])),
              lookahead=read('evidence/misul-lookahead-comparison.json'),
              fixedpoint=read('evidence/misul-fixedpoint-comparison.json'),
              fusion=read('evidence/misul-memory-fusion-comparison.json'))
for arm in result['fixedpoint']['runs'].values():
    assert arm['summary']['complete'] and arm['summary']['initial_model_sha256'] == storage['weight_sha256']
Path('evidence/misul-results.json').write_text(json.dumps(result, indent=2) + '\n')
macros = {'MisulValidationDelta': f'{100 * loss_change:+.2f}',
          'MisulTestDelta': f"{100 * result['precision_gate']['relative_test_loss_change']:+.2f}",
          'MisulReleaseBytes': str(result['storage']['complete_release_bytes'])}
for prefix, name in (('Standard', 'standard'), ('Fixed', 'fixedpoint')):
    arm = result['fixedpoint']['runs'][name]
    macros['Misul' + prefix + 'Flow'] = f"{100 * arm['summary']['flow']['8']['exact_accuracy']:.2f}"
    macros['Misul' + prefix + 'ValDelta'] = f"{100 * arm['summary']['relative_language_loss_change']:+.2f}"
    macros['Misul' + prefix + 'Minutes'] = f"{arm['guard']['elapsed_seconds'] / 60:.2f}"
for label, run in (('FP', fp8), ('BF', bf16)):
    evaluation, accounting = run['evaluation'], run['accounting']
    values = {
        'Validation': run['summary']['final_validation_loss'],
        'Test': evaluation['text_test']['loss'],
        'BitsPerByte': evaluation['text_test']['bits_per_byte'],
        'Minutes': accounting['actual_guarded_training_seconds'] / 60,
        'PhysicalGB': accounting['peak_physical_bytes'] / 1_000_000_000,
        'StreamHeld': 100 * evaluation['held_out']['streams']['answer_accuracy'],
        'StreamLong': 100 * evaluation['longer_lengths']['streams']['answer_accuracy'],
        'IdleThought': 100 * evaluation['held_out']['streams_with_idle_thought']['answer_accuracy'],
        'FlowLong': 100 * evaluation['longer_lengths']['flow']['8']['exact_accuracy'],
        'FlowNoFeedback': 100 * evaluation['held_out']['flow_without_feedback']['8']['exact_accuracy'],
    }
    values |= {f'Flow{word}': 100 * evaluation['held_out']['flow'][str(n)]['exact_accuracy']
               for n, word in ((1, 'One'), (2, 'Two'), (4, 'Four'), (8, 'Eight'))}
    for name, value in values.items():
        decimals = 4 if name in ('Validation', 'Test', 'BitsPerByte') else 3 if name == 'PhysicalGB' else 2
        macros[f'Misul{label}{name}'] = f'{value:.{decimals}f}'
    macros[f'Misul{label}Retries'] = str(accounting['observed_overflow_attempts'])
lines = ['% Generated from the verified complete precision pair; do not edit numbers by hand.']
lines += ['\\newcommand{\\' + name + '}{' + value + '}' for name, value in macros.items()]
lines += [r'\newif\ifmisulprecisionpassed',
          r'\misulprecisionpassedtrue' if result['precision_gate']['passed'] else r'\misulprecisionpassedfalse']
lines += [r'\newif\ifmisulfixedpointpassed',
          r'\misulfixedpointpassedtrue' if result['fixedpoint']['passed'] else r'\misulfixedpointpassedfalse']
Path('paper/misul_results_macros.tex').write_text('\n'.join(lines) + '\n')
with Path('evidence/misul-training-curve.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=['precision', 'step', 'elapsed_seconds', 'monitor_loss'])
    writer.writeheader()
    writer.writerows(fp8_curve + bf16_curve)

plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#687386', 'text.color': '#171E2B',
                     'axes.labelcolor': '#171E2B', 'xtick.color': '#687386',
                     'ytick.color': '#687386', 'grid.color': '#D7DCE2', 'grid.linewidth': .5})
figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.65), layout='constrained')
for run, curve, color, style, name in ((fp8, fp8_curve, '#0866C6', '-', 'FP8'),
                                      (bf16, bf16_curve, '#687386', '--', 'BF16 control')):
    axes[0].plot([p['step'] / 2 * 512 / 1_000_000 for p in curve], [p['monitor_loss'] for p in curve],
                 color=color, linestyle=style, linewidth=1.5, label=name)
axes[0].set(xlabel='Retained language target tokens (millions)', ylabel='Monitor loss (nats/token)', title='Learning from random initialization')
axes[0].legend(frameon=False, fontsize=8)
storage_mb = [result['storage']['tensor_bytes'] / 1_000_000,
              result['storage']['equal_parameter_bf16_bytes'] / 1_000_000]
bars = axes[1].bar(['FP8 + scales', 'Equal-parameter BF16'], storage_mb,
                   width=.55, color=['#0866C6', '#687386'])
axes[1].bar_label(bars, labels=[f'{v:.2f} MB' for v in storage_mb], padding=3, fontsize=8)
axes[1].set(ylabel='Counted learned tensor payload (MB)', title='Compact model storage', ylim=(0, max(storage_mb) * 1.17))
axes[1].tick_params(axis='x', labelsize=8)
for axis in axes:
    axis.grid(axis='y')
figures = Path('paper/figures')
figures.mkdir(exist_ok=True)
figure.savefig(figures / 'misul_results.pdf')
figure.savefig(figures / 'misul_results.png', dpi=180)
plt.close(figure)
print(json.dumps(dict(precision_gate=result['precision_gate'], storage=result['storage'],
                      fp8_accounting=fp8['accounting'], bf16_accounting=bf16['accounting']), indent=2))
