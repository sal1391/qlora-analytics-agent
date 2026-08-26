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

To put a number on it: every turn of one of these agents carries about 800
tokens of schema and business context before the user's question even starts,
and returns about 50 tokens of JSON. Multiply that by every question an
analytics team asks in a day.

Everything runs on generated data. I built SEAAD, the Synthetic Enterprise
Analytics Agent Dataset, specifically for this: a fictional company with a
fictional star-schema warehouse, and 1,092 instruction records covering
text-to-SQL, skill routing, clarification, and refusal of unsafe queries.
Every gold query was executed against the database before it was allowed into
the dataset, and the whole thing regenerates from a seed.

It worked, and it was cheap. QLoRA on Qwen2.5-3B, training half a percent of
its parameters in about 5 GB of VRAM on a single consumer GPU, took SQL
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

To put a number on it: every turn of one of these agents carries about 800
tokens of schema and business context before the user's question even starts,
and returns about 50 tokens of JSON. Multiply that by every question an
analytics team asks in a day.

I decided to find out, on my own time, on my own GPU.

Everything runs on generated data. I built SEAAD, the Synthetic Enterprise
Analytics Agent Dataset, for this project: a fictional company with a
fictional star-schema warehouse, and 1,092 instruction records covering
text-to-SQL, skill routing, clarification, and refusal of unsafe queries.
Every gold query was executed against the database before it made it into the
dataset, and the whole thing regenerates from a seed.

It worked better than I expected. QLoRA on Qwen2.5-3B, training half a percent
of its parameters in 5 GB of VRAM, went from 23% SQL execution accuracy (my
rule-based baseline) to 100% on held-out tests, and out-routed the rules as
well.

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

### Where the token numbers come from

The 800-in / 50-out figures are measured, not estimated: run over the 166
test-set records with the Qwen2.5 tokenizer, input averaged 806 tokens
(range 798-812) and output averaged 51 (range 34-101). Almost all of the
input is fixed overhead paid on every turn (schema DDL, business rules,
metric definitions); the user's actual question is a rounding error.

No dollar figures anywhere in the post, deliberately. The paper doesn't
contain a cost analysis, so a savings claim would be unsupported if anyone
opened the PDF looking for it. A proper throughput/cost benchmark is the
right thing to add in a v2.

### Witty privacy line, if you want it back

- "No real enterprise data was harmed in the making of this paper."
- "Every number in this paper comes from a company that doesn't exist."
