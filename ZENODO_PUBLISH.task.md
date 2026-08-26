# Zenodo publish walkthrough — progress tracker

Carlos drives; Claude navigates one step at a time. Update the checkbox when a
step is confirmed done. If the session dies, resume from the first unchecked
step.

## Phase A — Rebuild the PDF (Overleaf)

- [x] A1. Replace `main.tex` in Overleaf with the updated local copy
      (`latex/main.tex`, has results + FIU + no watermark)
- [x] A2. Upload the 5 regenerated figures into the Overleaf `figures/` folder
- [x] A3. Recompiled; Claude verified via text extraction: no DRAFT watermark,
      FIU affiliation, new results table, leakage note present
- [x] A4. Final PDF committed as `Salgado_2026_SEAAD_QLoRA_Analytics_Agents.pdf`
      and pushed (commit 7ce57db)

## Phase B — Zenodo record

- [x] B1. Signed in to Zenodo via GitHub
- [ ] B2. New upload: add the final PDF + repo zip (or GitHub release route)
- [ ] B3. Metadata: Publication → Preprint; title/authors/description from
      .zenodo.json; keywords; license CC-BY-4.0
- [ ] B4. Reserve DOI (before publishing) — copy it
- [ ] B5. (Optional) Put DOI into the PDF footnote, recompile, re-upload
- [ ] B6. Publish (permanent — everything above must be verified first)

## Phase C — Go public

- [ ] C1. Flip GitHub repo public
- [ ] C2. Add DOI badge + citation block to README; commit + push
- [ ] C3. LinkedIn post (LINKEDIN_POST.md) with the DOI link
- [ ] C4. Add paper to LinkedIn Publications section

## Notes / decisions

- Affiliation: FIU (set in paper + .zenodo.json)
- ORCID: optional but recommended — can be added to Zenodo before publishing
