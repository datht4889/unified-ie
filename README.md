# Unified IE

One generative model trained on several IE datasets at once. For now it's event detection only. NER data isn't picked yet, and the RE labels (TACRED, FewRel) aren't merged yet.

The training code is EventKD: a Qwen3-0.6B student learns from a LoRA-tuned Qwen3-4B teacher with CE + skewed forward KL + span-level representation distillation (`span_finetune.py`). No pseudo labels, no self-distillation.

## Data

Data lives on Hugging Face (private dataset `datht/unified-ie`), not in git. Pull it into `data/`:

```bash
huggingface-cli download datht/unified-ie --repo-type dataset --local-dir data
```

`data/general_domain/` holds ACE05, MAVEN, RAMS and GENEVA merged into one label set of 233 event types. Each row asks for one sentence's events. The prompt doesn't name the dataset or list the labels.

- `ee_v1/`: LDC labels written `level1.level2` (`conflict.attack`), FrameNet labels as MAVEN writes them (`Hostile_encounter`)
- `ee_v2/`: every label in lowercase words (`conflict attack`, `hostile encounter`)

The two versions have the same rows in the same order. Only the label strings differ.

`ee_label_map.tsv` maps every original label to `eval_label` (the label in its own dataset's scheme) and to its v1/v2 name. `label_list.md` lists the 233 labels and which datasets they come from. Each row also carries `dataset` and `gold_eval`, so predictions can be mapped back and scored per dataset.

Targets are `{"events": [[trigger, type], ...]}`. Arguments are left out because MAVEN and RAMS rows have none here.

## Run

```bash
bash install.sh
bash scripts/general_domain/process_data.sh v1          # tokenize into processed_data/
bash scripts/general_domain/sft_teacher_4B.sh v1        # LoRA SFT of the Qwen3-4B teacher
bash scripts/general_domain/eventkd_0.6B_4B.sh v1 <teacher_lora_dir>
```

Swap `v1` for `v2` to train on the other label format.
