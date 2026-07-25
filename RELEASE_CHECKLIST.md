# Zenodo release checklist

Goal: one citable Zenodo record (paper PDF + code archive) you can link from
LinkedIn and list on a grad-school application.

## 0. Decide first (5 minutes)

- [x] **Training done (2026-07-25).** All three Qwen2.5-3B QLoRA ablations
  trained and evaluated locally; paper updated with real numbers (r=32:
  95.8% routing, 100% SQL exec, 100% safety/JSON). Evaluation leakage bug
  found, fixed, and regression-tested before accepting results.
- [x] **Affiliation.** Done — set to Florida International University in
  `latex/main.tex` (author block + acknowledgment) and `.zenodo.json`.
  Employer continuation stays out of the paper until it's approved.
- [ ] **ORCID.** Free at https://orcid.org. Takes 5 minutes, and admissions
  committees and Zenodo both surface it. Add the ID to `.zenodo.json` as
  `"orcid": "0000-000X-XXXX-XXXX"` under your creator entry.

## 1. Rebuild the PDF (required)

The current PDF has a DRAFT watermark on every page. The watermark package has
been removed from `latex/main.tex` — recompile it (Overleaf or any pdflatex)
and replace the PDF. Rename it something clean, e.g.
`Salgado_2026_SEAAD_QLoRA_Analytics_Agents.pdf`.

If you update numbers after a training run, also update: abstract, Section VI
(results tables/figures), Conclusion, and `results/metrics.json`.

## 2. GitHub (recommended before Zenodo)

This folder is not a git repository yet.

```bash
git init
git add -A
git commit -m "SEAAD: QLoRA analytics agent — initial public release"
```

Then create the GitHub repo and push:

```bash
gh repo create qlora-analytics-agent --public --source . --push
```

Note: `data/raw/enterprise.duckdb` and adapter safetensors are binary but
small; check `.gitignore` isn't excluding anything the paper depends on.

## 3. Zenodo upload

1. Sign in at https://zenodo.org (GitHub login works and links your account).
2. Either:
   - **GitHub integration** (preferred): zenodo.org → GitHub settings page →
     flip the switch for the repo → create a GitHub release `v1.0.0`. Zenodo
     archives it automatically and reads `.zenodo.json` for metadata. Then
     upload the paper PDF to the same record (edit the draft before publishing),
     or
   - **Manual upload**: New upload → add the PDF + a zip of the repo.
3. Metadata (mostly prefilled by `.zenodo.json`):
   - Type: Publication → Preprint
   - License: CC-BY-4.0 for the record (code stays MIT via LICENSE in repo)
   - Related identifiers: GitHub repo URL ("is supplemented by this upload")
4. **Check "Reserve DOI" before publishing** so you can put the DOI inside the
   PDF (optional but nice: add `\thanks{DOI: 10.5281/zenodo.XXXXXXX}` to the
   title in main.tex, recompile, upload the final PDF).
5. Publish. Publishing is permanent — the record can be versioned but not
   deleted, so do the PDF/affiliation fixes first.

## 4. After the DOI exists

- [ ] Add a DOI badge + citation block to README.md
- [ ] CV / SoP line: Salgado, C. (2026). *Efficient Domain Adaptation of Open
      Language Models for Enterprise Analytics Agents Using LoRA and QLoRA.*
      Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
- [ ] LinkedIn post (draft in `LINKEDIN_POST.md`) — link the DOI, not a raw PDF
- [ ] Optional later: arXiv (cs.CL) once you have an endorsement; Zenodo DOI
      and arXiv coexist fine.
