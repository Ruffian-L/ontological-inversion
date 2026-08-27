# Zenodo concept DOI on the preprint

> Date: 2026-08-27
> Agent: Grok
> Repo: ontological-inversion

## Context

Jason published the preprint packet (PDF + zip) on Zenodo and quoted the
"cite all versions" blurb. GitHub–Zenodo is already flipped on.

Live record (fetched 2026-08-27):

- Concept DOI (all versions, always latest): `10.5281/zenodo.22126781`
- This version (v1): `10.5281/zenodo.22126782` / record https://zenodo.org/records/22126782
- Type: preprint, CC-BY-4.0
- Author on the record: Pham, Jason Van, ORCID 0009-0003-7212-6577
- Files: `Writing-Meaning-Between-Frozen-Models.pdf` md5 `29b7655ab7189fc6ce6f5d6cc22d9785`, `zenodo-writing-meaning-2026-08-27.zip` md5 `c50db961cad2c2b087d6cceaaa27187d`

No GitHub Releases. No second Zenodo hit for this GitHub URL besides the
preprint. Repo hooks visible via API: Snyk only. GitHub–Zenodo uses Zenodo's
GitHub app, not a classic webhook.

## What changed

Stamped the concept DOI on `paper/MANUSCRIPT.md`, repo README, `paper/README.md`,
`CITATION.cff`, and the Tectonic byline. Rebuilt the PDF. Did not replot
figures. Did not create a GitHub Release. Did not upload Zenodo v2.

## Hypothesis

People should cite `10.5281/zenodo.22126781`. A GitHub Release would mint a
*software* record with a different concept DOI; it would not become v2 of this
preprint.

## Next

Paper edits → new Zenodo version under the same concept. GitHub Release only
if Jason wants a separate code snapshot DOI.

Zenodo metadata nits still on v1 (not fixed here): keyword `onotlogical
inversion`; description GitHub link rendered with a space (`Ruffian- L`).
The related_identifier URL is the correct `Ruffian-L` form.
