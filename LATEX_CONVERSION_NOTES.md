# LaTeX Conversion Guide for NeurIPS 2026 E&D Track

## Template Setup

```latex
\documentclass{article}
\usepackage[eandd, nonanonymous]{neurips_2026}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{multirow}

\title{CogAttentionBench: Probing Cognitive Attention \\
       Mechanisms in Frontier AI Models}

\author{
  Zihan Zeng \\
  BIBS International Bilingual School \\
  Chengdu, China \\
  \texttt{corazeng@outlook.com}
}
```

## Key Formatting Notes

1. **Page limit**: 9 pages of content (references + appendix unlimited)
2. **Review mode**: Single-blind (author names visible)
3. **Required package**: `neurips_2026` with `eandd` and `nonanonymous` options
4. **Required**: Croissant metadata file (`croissant_metadata.json`) — already created

## Conversion Checklist

- [ ] Download NeurIPS 2026 LaTeX template from OpenReview
- [ ] Copy abstract, all sections, references
- [ ] Format Table 1 with `booktabs` (`\toprule`, `\midrule`, `\bottomrule`)
- [ ] Convert MFA equation to LaTeX: `$F_{att}(r) = S/r^2$`
- [ ] Add `\bibliography{references}` with BibTeX
- [ ] Verify page count ≤ 9 (excluding references + appendix)
- [ ] Add Croissant metadata reference
- [ ] Compile and check for formatting issues

## Submission Steps

1. **May 4**: Submit abstract on OpenReview
2. **May 6**: Upload full paper PDF + supplementary + Croissant metadata
3. Upload to OpenReview for NeurIPS 2026 Datasets and Benchmarks Track

## Files Ready

- `neurips_cogattentionbench.md` — Full paper in Markdown (6970 words)
- `croissant_metadata.json` — Dataset metadata (Croissant format)
- GitHub repo: https://github.com/Lumen-Nox/CogAttentionBench
