#! /bin/bash
# LoRA SFT of the Qwen3-4B teacher on sentence-level relation extraction, one teacher per dataset
# (TACRED and FewRel labels are not merged). The prompt gives only the sentence: the model writes
# every (subject, relation, object) triple, no entity pair is given.
# usage: bash scripts/general_domain/sft_teacher_4B_re.sh            # TACRED, then FewRel
#        bash scripts/general_domain/sft_teacher_4B_re.sh fewrel     # one of them
for DS in ${*:-tacred fewrel}; do
    bash scripts/general_domain/sft_teacher_4B.sh re_${DS}
done
