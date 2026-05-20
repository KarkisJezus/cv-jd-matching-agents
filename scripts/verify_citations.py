"""Greitas patikrinimas: ar visi \\cite{} raktai Sablonas.tex turi atitikmenis bibliografija.bib.

Naudojimas:
    python scripts/verify_citations.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent
tex_text = (REPO / "thesis" / "Sablonas.tex").read_text(encoding="utf-8")
bib_text = (REPO / "thesis" / "bibliografija.bib").read_text(encoding="utf-8")

cite_keys: set[str] = set()
for match in re.finditer(r"\\cite\{([^}]+)\}", tex_text):
    for key in match.group(1).split(","):
        cite_keys.add(key.strip())

bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_text))

missing = cite_keys - bib_keys
unused = bib_keys - cite_keys

print(f"Cited keys in Sablonas.tex: {len(cite_keys)}")
print(f"Defined keys in bibliografija.bib: {len(bib_keys)}")
print(f"Missing (cited but not defined): {len(missing)}")
for key in sorted(missing):
    print(f"  ! {key}")
print(f"Unused (defined but not cited): {len(unused)}")
for key in sorted(unused):
    print(f"  - {key}")
