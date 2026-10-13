# Releasing the companion

Run these commands from the companion-code root. The release tools live in
this tree and require Python 3.10–3.12 and Make. They do not require the
manuscript, author evidence archive, or upstream checkouts.

## Inventory

`release_inventory.json` classifies each reviewed file as included or excluded.
New files must be reviewed and registered before export. The exclusions retain
author instructions, extended investigations, and unrun study protocols in the
source checkout; they may be absent from a standalone release. Known local
environment and cache directories are ignored. All other unclassified files,
missing included files, symbolic links, and documentation links to omitted
files block export.

```sh
make release-inventory-check
make smoke
```

The smoke suite runs all `test_*.py` files in registered chapter directories,
including additional implementation and fixture tests. Optional upstream-source
tests report skips when their separately configured source checkouts are absent.
CUDA compilation and the Chapter 29 browser evaluator remain optional checks
with their requirements documented in their chapter READMEs.

For the complete local gate with the optional pinned quality tools installed:

```sh
poetry install --with quality
poetry run make quality-checks
```

The project has no Python runtime dependencies. Optional quality tool versions
are pinned in `pyproject.toml`; no lockfile is needed for `make smoke`.

## Local candidate

Choose a new directory outside this tree. The command refuses an existing
destination, checks the inventory and release hygiene, and copies only included
files. It does not create a GitHub repository, commit, push, or publish anything.

```sh
python3 -B -m _maintenance.release --output ../kimi-code-release
make -C ../kimi-code-release smoke
```

This command replaces the historical exporter in the author book tooling.
Release-policy changes are maintained entirely in the companion tree.

`release-manifest.json` records the source Git commit when Git is available and
the SHA-256 hash of every exported file. The commit supplies provenance; the
hashes identify the actual bytes, including any uncommitted source changes.
The exporter rejects source changes during copying. Before publication, verify
the candidate's hashes from its root:

```sh
python3 -B -m _maintenance.release --verify
```

`make smoke` also verifies these hashes when a release manifest is present.
After editing a candidate, re-export the reviewed source to a new directory
instead of manually updating its hashes. Publish the contents of the verified
candidate as the public repository root and choose an edition tag there.
Copying the complete book checkout or the unfiltered `code/` tree would bypass
the reviewed exclusions.
