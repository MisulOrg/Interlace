# Interlace

A shared architecture for language and iterative computation, developed by
Deyan Todorov at Misul.

Interlace combines bounded rational feed-forward features, content-selected
attention and gated delta memory in a backbone that reuses its layers across
successive computation steps. A single checkpoint supports text generation,
synchronous input/thought/output streams and experimental Flow-style refinement.

The 30M-parameter comparison measures 9.86% lower language-test loss and 65.7%
less training execution time to a fixed validation-monitor target than a
parameter-matched dense Transformer. Longer-program stream accuracy is
80.47%, compared with 4.69% for the dense control.

[Read the paper and system card (PDF)](paper/interlace.pdf) ·
[Technical supplement](TECHNICAL-SUPPLEMENT.md) ·
[Apache-2.0 license](LICENSE) · [Citation](CITATION.cff)

## Architecture

Interlace brings five mechanisms into one trainable backbone:

- Bounded rational features provide nonlinear feed-forward transformations.
- Content-selected attention connects each row to selected earlier rows.
- Gated delta memory carries and updates a recurrent state.
- Shared depth applies the same backbone repeatedly, with learned loop identifiers.
- Synchronous streams represent inputs, intermediate states and outputs in separate roles.

The text interface predicts the next token. The stream interface advances all
three roles together and learns intermediate states on prefix addition modulo
ten. Flow revises a candidate sequence over repeated passes. All three
interfaces share the released model's 29,814,640 parameters.

## Results

The architecture comparison uses BF16 for both models, random initialization,
seed 1001, identical data order, optimizer and task mixture, and 16,384 updates.
Interlace has 29,814,640 parameters; the dense control has 29,921,280. Both use
the same task interfaces and process 4,194,304 language target tokens.

| Measurement | Interlace BF16 | Dense BF16 |
| --- | ---: | ---: |
| WikiText-2 test loss, nats/token | 4.7339 | 5.2516 |
| Time to 5.5-nat validation-monitor target | 276.52 s | 806.91 s |
| Stream answer accuracy, held-out lengths 2–6 | 100.00% | 100.00% |
| Stream answer accuracy, longer lengths 8–10 | 80.47% | 4.69% |
| Total training execution | 22.03 min | 22.26 min |

These are single-seed results under a common training recipe. Time to target
uses the first observed crossing on a fixed 32-window validation monitor;
execution time includes compilation, data handling, validation, checkpoints
and retries. The [paper and system card](paper/interlace.pdf) report the full
method, inference-latency tradeoff, Flow results and evaluation scope.

## Quick start

The MLX reference implementation runs on Apple Silicon with Python 3.12 or newer.
The repository includes the Interlace-30M checkpoint and tokenizer.

```sh
git clone https://github.com/MisulOrg/Interlace.git
cd Interlace
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Generate text:

```sh
.venv/bin/python run.py text --prompt 'The city of' --tokens 64
```

Run synchronous streams and inspect the intermediate states:

```sh
.venv/bin/python run.py streams --digits '2,5,3,7'
```

Try experimental iterative refinement:

```sh
.venv/bin/python run.py flow --digits '2,5,3,7' --flow-steps 8
```

[Examples](examples/) contain output from all three modes. The launcher applies
resource limits automatically; configuration and runtime requirements are
specified in the [technical supplement](TECHNICAL-SUPPLEMENT.md).

## Interlace-30M checkpoint

The checkpoint was trained from random initialization and contains all three
interfaces in one parameter set. Packed weights occupy 30.76 MB, with FP8
weights and projection operands and wider working tensors.

| Released checkpoint measurement | Result |
| --- | ---: |
| WikiText-2 test loss | 4.7602 nats/token |
| Stream answer accuracy, held-out lengths 2–6 | 100.00% |
| Stream answer accuracy, longer lengths 8–10 | 92.97% |
| Packed weight file | 30.76 MB |

The released FP8 checkpoint is distinct from the BF16 architecture comparison
above. Its test loss is 0.56% above the BF16 checkpoint of the same architecture.
The [system card](paper/interlace.pdf) describes intended research use and
current text and refinement capabilities.

## Verify and reproduce

The repository includes numerical tests, training and evaluation programs,
registered experiment protocols, model evaluations and timing records.
`BUNDLE-MANIFEST.json` records file sizes and SHA-256 digests.

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
PYTHONPATH=src .venv/bin/python -m transformermodel.safe_run \
  --report test-guard.json --seconds 180 -- \
  .venv/bin/python -m pytest tests -q
```

The [technical supplement](TECHNICAL-SUPPLEMENT.md) gives corpus preparation,
training and evaluation details. Training data are downloaded separately.
The Python API is `transformermodel.interlace.Interlace`.

## License and citation

Copyright 2026 Deyan Todorov. Interlace is licensed under Apache-2.0; see
[LICENSE](LICENSE) and [NOTICE](NOTICE). Use [CITATION.cff](CITATION.cff) to cite
the work in research. External datasets and dependencies retain their own licenses.

## References

1. A. Dhurandhar et al. [CoFrGeNet: Continued Fraction Architectures for Language Generation](https://arxiv.org/abs/2601.21766). 2026.
2. J. Zhang et al. [Rational ANOVA Networks](https://arxiv.org/abs/2602.04006). 2026.
3. Kimi Team. [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692). 2025.
4. A. Helbling et al. [Flow Reasoning Models: Turning Flows Into Efficient Recurrent Reasoners](https://arxiv.org/abs/2606.29150). 2026.
5. N. Amsel et al. [The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm](https://arxiv.org/abs/2505.16932). 2025.
6. Z. Li et al. [NorMuon: Making Muon more efficient and scalable](https://arxiv.org/abs/2510.05491). 2025.
7. N. Shazeer. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202). 2020.
8. Salesforce. [WikiText-2 raw v1](https://huggingface.co/datasets/Salesforce/wikitext). Pinned revision and preparation records accompany the reproducibility package.
9. D. Todorov. [Monodratic: A Sparse Attention Architecture with Learned Product-Hash Routing](https://github.com/MisulOrg/Monodratic/blob/main/output/pdf/monodratic_proof.pdf). Technical report, 2026.
