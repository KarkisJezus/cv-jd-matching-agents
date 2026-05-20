"""Auditas 1 sk. atitiktis vadovo kritikai."""
from __future__ import annotations
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

tex = Path("thesis/Sablonas.tex").read_text(encoding="utf-8")

# Find chapter 1 boundaries — robust to other \section calls
m = re.search(r"\\section\{Literatūros analizė\}(.*?)\\section\{", tex, re.DOTALL)
if not m:
    print("Nerasta 1 sk.")
    sys.exit(1)
ch1 = m.group(1)

# Strip LaTeX commands for word count
text_only = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})*", " ", ch1)
text_only = re.sub(r"%.*", "", text_only)
text_only = re.sub(r"\s+", " ", text_only)
words = text_only.split()
print(f"=== 1 SK. APIMTIES METRIKOS ===")
print(f"Žodžių skaičius (apytiksliai): {len(words)}")
print(f"Eilučių skaičius: {ch1.count(chr(10))}")
print(f"Simbolių skaičius: {len(ch1):,}")
print()

# Citations in chapter 1 only
ch1_citations: set[str] = set()
for match in re.finditer(r"\\cite\{([^}]+)\}", ch1):
    for key in match.group(1).split(","):
        ch1_citations.add(key.strip())
print(f"=== 1 SK. ŠALTINIAI ===")
print(f"Unikalūs cituojami šaltiniai 1 sk.: {len(ch1_citations)}")
for k in sorted(ch1_citations):
    print(f"  • {k}")
print()

# Subsections
print(f"=== 1 SK. STRUKTŪRA ===")
subsections = re.findall(r"\\subsection\{([^}]+)\}", ch1)
print(f"Poskyriai ({len(subsections)}):")
for s in subsections:
    print(f"  - {s}")
print()

# Tables and figures
tables = re.findall(r"\\caption\{([^}]+)\}", ch1)
print(f"=== 1 SK. VIZUALIZACIJOS ===")
print(f"Lentelės/paveikslai: {len(tables)}")
for t in tables:
    print(f"  · {t[:120]}")
print()

# Per-subsection word count
print(f"=== 1 SK. APIMTIS PER POSKYRIUS ===")
parts = re.split(r"\\subsection\{[^}]+\}", ch1)
# parts[0] is the intro, parts[1+] are subsections
intro_words = len(re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})*", " ", parts[0]).split())
print(f"  Įvado žodžiai: {intro_words}")
for name, part in zip(subsections, parts[1:]):
    sub_words = len(re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})*", " ", part).split())
    print(f"  {name}: {sub_words} žodžių")
