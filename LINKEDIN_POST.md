# LinkedIn post drafts

Two tone options below. Both use the real origin story (a cost/architecture
question from building BI agents, not a course project), reference the day job
generically, and leave out the evaluation-bug story.

Audience: AI company hiring managers and recruiters, plus practitioners
building agent systems.

---

## Version A — confident practitioner

I published my first research paper.

The question behind it came from work. I build BI agents, and the default
architecture puts a large hosted model at the center of everything: routing
each question to the right skill, calling tools, writing the SQL. Every turn
pays frontier-model prices. I wanted to know whether that was actually
necessary or just the habit everyone inherited. Could a small open-weight
model be fine-tuned to run skills and use tools well enough to take that
traffic, at a fraction of the cost?

No real enterprise data was harmed in the making of this paper. That was
requirement number one, and it wasn't a joke: everything runs on SEAAD, a
fully synthetic benchmark I built for this. Fictional company, fictional star
schema, 1,092 validated instruction records covering text-to-SQL, skill
routing, and refusal of unsafe queries. Anyone can regenerate all of it from
a seed.

Short answer: yes, and cheaply. QLoRA on Qwen2.5-3B, training half a percent
of its parameters in about 5 GB of VRAM on a single consumer GPU, took SQL
execution accuracy from 23% (my hand-written rule baseline) to 100% on the
held-out test set. It beat the rules at skill routing too.

The finding I didn't expect ran the other direction. The untouched base model
scored 0% on routing while emitting valid JSON almost every time. It could not
infer the agent's internal skill taxonomy from the prompt no matter how
well-formed its output was. The contract has to be learned, not assumed. If
you build skill-based agents, that's the part worth taking away.

Code, dataset generators, and training scripts are public and reproduce on one
consumer GPU or a few dollars of Colab credits.

https://doi.org/10.5281/zenodo.22117133

---

## Version B — milestone-forward and personal

I published my first research paper today.

It started with a question I couldn't let go of. I build BI agents, and nearly
everyone builds them the same way: a large hosted model sits at the center and
handles every turn, routing the question to the right skill, picking tools,
writing SQL. That gets expensive fast, and I kept wondering whether it was
necessary or just convention. Why couldn't a much smaller model be trained to
do the skill-running and tool-using at a fraction of the cost?

I decided to find out, on my own time, on my own GPU.

No real enterprise data was harmed in the making of this paper. That was rule
one, and I meant it: I built SEAAD, a fully synthetic benchmark, so the whole
thing could be public. Fictional company, fictional star schema, 1,092
validated records covering text-to-SQL, skill routing, and refusal of unsafe
queries. Regenerable from a seed by anyone who wants to check my work.

It worked better than I expected. QLoRA on Qwen2.5-3B, training half a percent
of its parameters in 5 GB of VRAM, went from 23% SQL execution accuracy (my
rule-based baseline) to 100% on held-out tests, and out-routed the rules as
well.

The result that stuck with me was a different one. The base model, untouched,
scored 0% at routing while producing valid JSON nearly every time. It simply
could not guess the agent's internal skill taxonomy. The contract has to be
learned, not assumed.

Building the benchmark, training the models, writing it up, and publishing it
taught me more than I expected going in. All of it is public and reproducible
on a single consumer GPU.

https://doi.org/10.5281/zenodo.22117133

---

## Notes

- Post the DOI link, not a PDF attachment. The DOI is permanent and external
  links carry further than uploaded documents.
- One or two hashtags at most (#MachineLearning #LLM) or none.
- Add the paper to your LinkedIn Publications section with the same DOI.
- Neither version says or implies you are looking for a role. If a recruiter
  reads it, the work speaks; if a colleague reads it, it's a side project you
  published.

### Other phrasings for the privacy line, if you want a different flavor

- "No real enterprise data was harmed in the making of this paper."
- "No production data was harmed in the making of this research. Not one row."
- "Zero rows of real customer data were involved, which was the whole point."
- "Every number in this paper comes from a company that doesn't exist."
