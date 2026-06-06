# Architecture Transfer Matrix

Use this matrix to convert multimodal research into notebook experiments.

| Source Pattern | Translate To SA3/SAME | Likely Altitude | Measurement | Caution |
|---|---|---|---|---|
| Diffusion/flow inversion | Flow prompt/state scoring over `z_t`, timestep/logSNR, velocity convention | root rows plus `procedures/flow_scoring.py` | flow loss rows, attribution, decoded audition | velocity sign and timestep parameterization must be explicit |
| Attention/control maps | Prompt token attribution, residual steering, or condition ablation | root prompt rows, residual adapter, residual procedures | token loss deltas, residual probes, descriptor deltas | SA3 may not expose image-like spatial attention semantics |
| Video temporal coherence | Loop lab, periodicity probes, control lanes, bridge search | root looping/periodic/control lanes plus evidence | loop metrics, autocorrelation, player loop audition | audio time structure is multi-scale and not just frame smoothness |
| Representation probing | Linear control probes, residual feature basis, geometry reports | root measurement modules | R2, heldout rows, source-preservation/novelty | probe accuracy is not intervention success |
| Adapter/LoRA/finetune methods | Underfit external training, notebook comparison of exports | external plus evidence/comparison procedures | descriptors, flow prompt scores, memory similarity, listening notes | keep training outside this repo unless notebook consumes artifacts |
| Retrieval-augmented generation | Latent memory index, curriculum, donor/source selection | root memory/composition modules | nearest rows, geometry distance, descriptor match | retrieval can accidentally reward copying |
| Preference/reward optimization | Promote/drop criteria, annotation-weighted selection | evidence and ledger, not fake reward training | ledger decisions, descriptor/listening agreement | avoid fake scalar rewards without enough listening data |
