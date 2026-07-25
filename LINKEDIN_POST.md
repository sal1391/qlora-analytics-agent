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

The results taught me two things. First: fine-tuning works. QLoRA on
Qwen2.5-3B (training half a percent of its parameters, 5 GB of VRAM on my
laptop's GPU) took SQL execution accuracy from 23% for my hand-written rule
baseline to 100% on the held-out test set, and beat the rules at skill
routing too. Second, and more interesting: the untouched base model scored
0% at routing even though it output valid JSON almost every time. It could
not guess my agent's internal skill taxonomy from the prompt. The contract
has to be learned, not assumed.

Bonus lesson: my first evaluation run showed the base model scoring 100%,
which is how I learned it was reading the answer key out of its own prompt.
If your baseline looks too good, it is. Fixed, regression-tested, re-run.

Code, dataset generators, and training scripts are all public and run on a
single consumer GPU or a few dollars of Colab credits.

Paper + code: https://doi.org/10.5281/zenodo.XXXXXXX

---

## Shorter alternative

New paper on Zenodo: I built a fully synthetic benchmark (SEAAD) for
enterprise analytics agents and fine-tuned Qwen2.5-3B with QLoRA for skill
routing and safe SQL generation. Training 0.5% of parameters in 5 GB of VRAM
took SQL execution accuracy from a 23% rule-baseline to 100% in-distribution,
while the un-tuned base model couldn't route at all (0%) despite emitting
valid JSON. All code and data generators are public and reproducible on one
consumer GPU.

https://doi.org/10.5281/zenodo.XXXXXXX

## Notes

- Post the DOI, not a PDF attachment; LinkedIn deprioritizes documents less
  than external links these days, and the DOI is the permanent one.
- Skip hashtag walls; one or two at most (#MachineLearning #LLM) or none.
- Good time to update your LinkedIn Publications section with the same DOI.
