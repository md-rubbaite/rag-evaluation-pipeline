# Security and privacy notes

- **No network access.** Evaluation runs entirely offline; no document, question or answer
  leaves the machine or CI runner. This matters for regulated corpora (financial, health).
- **No runtime dependencies.** The package uses only the Python standard library, which keeps
  the supply-chain surface to the interpreter itself. Development tools (pytest, ruff) are not
  installed in production use.
- **Untrusted input.** JSONL and TOML are parsed with the standard library (`json`, `tomllib`);
  nothing is evaluated or imported from the data files. Malformed lines fail with the file and
  line number.
- **Reports may contain corpus text.** Unsupported sentences are copied into `report.md` and
  `report.json`. Treat reports with the same classification as the corpus, and restrict who can
  download CI artifacts for private data.
- **Use synthetic or approved data in CI.** Do not commit real customer data to evaluation
  sets; build them from synthetic, anonymised or explicitly approved material.
