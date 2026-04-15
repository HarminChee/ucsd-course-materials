# Publishing Audit

Generated on 2026-04-15 from:

`/Users/harmin/Desktop/windows/UCSD/Course`

Target public workspace:

`/Users/harmin/Desktop/BenchInject/ucsd-course-materials-full-open`

## Publishing Premise

This version includes official course PDFs, assignments, quizzes/exams, and solution PDFs under the user's statement that course instructors explicitly recommend public GitHub publishing of these materials.

## Source Folders

Original course folders:

| Folder | Original size |
| --- | ---: |
| CSE227 | 17 MB |
| CSE256 | 4.8 GB |
| CSE258 | 2.1 GB |
| ECE251C | 2.8 MB |
| ECE253 | 83 MB |
| ECE269 | 1.3 GB |
| ECE271A | 19 MB |
| ECE285 | 81 MB |

## Public Subset

After removing virtual environments, caches, nested Git metadata, and `ECE271A`, the full public workspace contains:

- 7 course folders.
- 326 copied course files before generated repository docs.
- About 2.4 GB before Git LFS pointer conversion.
- 7 files over 95 MB, tracked with Git LFS.

## Excluded Rules

The copy excludes:

- `ECE271A/`
- `.git/`
- `.venv*/`
- `venv/`
- `__pycache__/`
- `.ipynb_checkpoints/`
- `node_modules/`
- `build/`
- `dist/`
- `.pytest_cache/`
- `.mypy_cache/`
- `.DS_Store`

## Large Files

Large files are tracked through Git LFS so GitHub can accept the repository:

- `*.mat`
- `*.tiff`
- `*.tsv.gz`
- `*.zip`
- `CSE258/dataset/*.json`

## Privacy / Secret Scan

The generated repository is checked for obvious private-key and credential filenames. Text-based scans are also run for direct email exposure and obvious secret/token patterns. Binary PDFs, DOCX, PPTX, and notebooks may still require manual spot review because embedded metadata and rendered content are harder to audit mechanically.
