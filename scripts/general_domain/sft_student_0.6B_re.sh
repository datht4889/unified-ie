#! /bin/bash
# SFT of the Qwen3-0.6B student (full fine-tuning, CE only) on sentence-level relation extraction,
# one model per dataset like the RE teachers (TACRED and FewRel labels are not merged).
# usage: bash scripts/general_domain/sft_student_0.6B_re.sh            # TACRED, then FewRel
#        bash scripts/general_domain/sft_student_0.6B_re.sh fewrel     # one of them
for DS in ${*:-tacred fewrel}; do
    bash scripts/general_domain/sft_student_0.6B.sh re_${DS}
done
