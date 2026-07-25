# LinkedIn post draft

Paste the DOI link once Zenodo issues it. If you run the Qwen2.5-3B training
first, update the numbers in paragraph four (marked below).

---

I just put my first research paper on Zenodo.

It started as a course project and turned into something I couldn't put down:
can you fine-tune a small open-weight model to do the unglamorous jobs inside
an enterprise BI agent, like routing a question to the right skill and writing
SQL that is actually safe to run?

Since I work with real warehouse data all day, rule number one was that none
of it goes anywhere near the paper. So I built SEAAD, a fully synthetic
benchmark. Fictional company, fictional star schema, 1,092 validated
instruction records covering text-to-SQL, skill routing, and refusal of
unsafe queries. Anyone can regenerate the whole thing from a seed.

The results surprised me. LoRA took a small router model from 27% to 61.5%
accuracy while training just 1.3% of its parameters. But my hand-written rule
baseline still won at routing (92.8%) and was perfect on safety refusals.
Where the rules fell apart was generating SQL: 23% execution accuracy. That
gap is the whole story. Deterministic guardrails where they're strong, learned
models where they're not.

[IF TRAINING RUN COMPLETED, replace the paragraph above with the QLoRA
Qwen2.5-3B numbers: rank ablation r=8/16/32, SQL execution accuracy vs. the
template baseline, VRAM footprint on the 5090.]

Code, dataset generators, and training scripts are all public and run on a
single consumer GPU or a few dollars of Colab credits.

Paper + code: https://doi.org/10.5281/zenodo.XXXXXXX

---

## Shorter alternative

New paper on Zenodo: I built a fully synthetic benchmark (SEAAD) for
enterprise analytics agents and tested whether LoRA/QLoRA can specialize
small open models for skill routing and safe SQL generation. LoRA moved
routing accuracy from 27% to 61.5% training 1.3% of parameters, and a
rule-based baseline showed exactly where learned models are and aren't
needed. All code and data generators are public and reproducible on one
consumer GPU.

https://doi.org/10.5281/zenodo.XXXXXXX

## Notes

- Post the DOI, not a PDF attachment; LinkedIn deprioritizes documents less
  than external links these days, and the DOI is the permanent one.
- Skip hashtag walls; one or two at most (#MachineLearning #LLM) or none.
- Good time to update your LinkedIn Publications section with the same DOI.
