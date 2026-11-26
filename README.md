# S4D batch / learning rate scaling

Reproduce the report's fixed-token experiment: ordinary LR 0.01, dynamics LR 0.001; batches 64/256/512; linear/square-root ordinary scaling crossed with unchanged/square-root/linear dynamics scaling; 16 epochs, seeds 11/22/33 (39 runs). Tiny Shakespeare is included.

```bash
python -m pip install -r requirements.txt
python train.py
python summarize.py
```

Run one policy plus its reference:

```bash
python train.py --seeds 11 --batches 64 512 --weight-rules sqrt --dynamics-rules linear --output subset
python summarize.py --results subset
```

Requires Python 3.12 and a CUDA GPU. Outputs: per-run configuration, loss curves, response profiles, final weights, and aggregate metrics. Completed runs are skipped when rerunning the same command.

Model adapted from [state-spaces/s4](https://github.com/state-spaces/s4/tree/e757cef57d89e448c413de7325ed5601aceaac13): required classes extracted, comments/docstrings removed, character-language-model forward added. Apache-2.0 license retained in LICENSE.
