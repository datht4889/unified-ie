# Unified IE

One generative model trained on several IE datasets at once. For now it's event extraction only. NER data isn't picked yet, and the RE labels (TACRED, FewRel) aren't merged yet.

The training code is EventKD: a Qwen3-0.6B student learns from a LoRA-tuned Qwen3-4B teacher with CE + skewed forward KL + span-level representation distillation (`span_finetune.py`). No pseudo labels, no self-distillation.

## Data

Data lives on Hugging Face (private dataset `datht/unified-ie`), not in git. Pull it into `data/`:

```bash
huggingface-cli download datht/unified-ie --repo-type dataset --local-dir data
```

`data/general_domain/` holds ACE05, MAVEN, RAMS and GENEVA merged into one label set of 233 event types. The prompt doesn't name the dataset or list the labels.

| folder | task | datasets |
|---|---|---|
| `ee_v1/`, `ee_v2/` | event detection: `{"events": [[trigger, type], ...]}` | ACE, MAVEN, RAMS, GENEVA |
| `ee_arg_v1/`, `ee_arg_v2/` | event extraction with arguments: `[[trigger, type, [[argument, role], ...]], ...]` | ACE, GENEVA |

v1 writes LDC labels as `level1.level2` (`conflict.attack`) and FrameNet labels as MAVEN does (`Hostile_encounter`). v2 writes every label (and role) in lowercase words (`conflict attack`, `hostile encounter`). Same rows, same order, only the label strings differ.

Arguments are a separate training run because MAVEN and RAMS rows have none: mixed in, they'd teach "no arguments" for the types those datasets share with ACE and GENEVA.

Each row carries `dataset` and `gold_eval` (gold in the dataset's own labels). `dev.jsonl` keeps at most 300 rows per dataset for checkpoint selection, `dev_full.jsonl` has all of them. `ee_label_map.tsv` maps every original label to its merged v1/v2 name, and `label_list.md` lists the 233 labels.

## Training

```bash
bash install.sh
bash scripts/general_domain/process_data.sh ee_v1                  # tokenize into processed_data/
bash scripts/general_domain/sft_teacher_4B.sh ee_v1                # LoRA SFT of the Qwen3-4B teacher
bash scripts/general_domain/eventkd_0.6B_4B.sh ee_v1 <teacher_lora_dir>
```

Swap `ee_v1` for `ee_v2`, `ee_arg_v1` or `ee_arg_v2`.

Things that differ from the original EventKD scripts:

- `--sample-alpha 0.5`: rows of dataset d are drawn with probability prop. to n_d^0.5 (temperature 2, as in T5 and BERT-and-PALs). An epoch is still one pass worth of rows, so MAVEN (70% of the rows) gets about 48% of the steps and GENEVA goes from 5% to 13%.
- `--label-map`: dev/test are also scored per dataset in its own labels (`by_dataset` in the log). A predicted label the row's dataset doesn't use is dropped, and RAMS is scored on its 38 level-2 types. Pick the checkpoint on `by_dataset.macro_trigger_f1` of dev.
- Decoding is greedy (no `--do-sample`). `do_sample` is passed straight to `generate()`, because transformers otherwise fills it from Qwen3's `generation_config.json`, which samples.
- `process_data.py` drops train/dev rows longer than `--max-prompt-length` / `--max-length` instead of cutting them (10 train rows, 1 dev row). Test rows are never dropped.
