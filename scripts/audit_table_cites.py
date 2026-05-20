"""Patikrina, ar visi lentelėse cituojami šaltiniai cituojami ir aplinkinėje pastraipoje."""
from __future__ import annotations
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

tex = Path("thesis/Sablonas.tex").read_text(encoding="utf-8")

# Find chapter 1
m = re.search(r"\\section\{Literatūros analizė\}(.*?)\\section\{", tex, re.DOTALL)
ch1 = m.group(1) if m else ""

# Find all tables in chapter 1
tables = list(re.finditer(r"\\begin\{table\}.*?\\end\{table\}", ch1, re.DOTALL))
print(f"Lentelės 1 sk.: {len(tables)}\n")

for i, t in enumerate(tables, 1):
    label_m = re.search(r"\\label\{([^}]+)\}", t.group(0))
    label = label_m.group(1) if label_m else "no label"
    cites_in_table = re.findall(r"\\cite\{([^}]+)\}", t.group(0))
    keys_in_table: set[str] = set()
    for c in cites_in_table:
        for k in c.split(","):
            keys_in_table.add(k.strip())

    prose = ch1[:t.start()] + ch1[t.end():]

    print(f"{i}. {label}: {len(keys_in_table)} citatos lentelėje")
    for k in sorted(keys_in_table):
        cited_in_prose = bool(re.search(r"\\cite\{[^}]*\b" + re.escape(k) + r"\b[^}]*\}", prose))
        marker = "✓ taip pat tekste" if cited_in_prose else "✗ TIK lentelėje (PAVOJINGA šalinti)"
        print(f"   {k}: {marker}")
    print()
