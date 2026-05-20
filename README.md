# CV ir darbo skelbimų semantinio atitikimo agentinė sistema

VU MIF bakalauro darbo prototipas — daugiaagentė sistema, kuri vertina gyvenimo aprašymų (CV) ir darbo skelbimų semantinį atitikimą naudodama didžiuosius kalbos modelius (LLM), Sentence-BERT embedding'us ir ESCO ontologiją.

> 📄 **Susijęs bakalauro darbas:** *„Autonominė daugiagentė sistema CV ir darbo skelbimų semantiniam atitikimui ir sprendimų priėmimui"*, Vilniaus universitetas, Matematikos ir informatikos fakultetas, 2026 m.

## Architektūra

Sistema remiasi **lentos (angl. blackboard) paradigma** — visi agentai bendrauja per vieną bendros būsenos objektą (`SharedContext`), į kurį kiekvienas rašo savo rezultatą ir iš kurio skaito ankstesnių agentų išvestis. Detalus architektūros aprašymas — `architecture.md` faile.

### 8 agentai

| Agentas | Vaidmuo | Įvestis | Išvestis |
|---|---|---|---|
| `cv_profiling` | Struktūrinis kandidato profilis iš CV teksto | Žalias CV tekstas | Įgūdžiai, patirtis, vyresniškumas |
| `jd_profiling` | Idealaus kandidato profilis + ESCO vaidmens klasifikacija | Darbo skelbimo tekstas | Reikalavimai, ESCO vaidmuo |
| `matching` | Kosinusinis panašumas + įgūdžių padengimas (be LLM iškvietimo) | Abu profiliai | Panašumo balai |
| `reasoning` | Stiprybių/trūkumų analizė ir pradinis balas | Profiliai + panašumai | Struktūrinis argumentavimas |
| `reflection` | Argumentavimo nuoseklumo tikrinimas; grąžina perskaičiuoti aptikus prieštaravimą | Samprotavimo išvestis | Nuoseklumo verdiktas |
| `decision` | Galutinis verdiktas (rekomendacijos kategorija + balas 0–100) | Visas kontekstas | Sprendimas |
| `memory_retrieval` | Panašiausių praeities atvejų paieška *(tik scenarijus C)* | Atminties bankas | 5 panašiausi įrašai |
| `calibration` | Pradinio balo peržiūra remiantis atminties įrašais *(tik scenarijus C)* | Pirmasis balas + atmintis | Peržiūrėtas balas |

Nepriklausomas **LLM-teisėjas** (DeepSeek-V3.2) atlieka aklą auditą — vertina sistemos sprendimus ir šaltinio etiketes, nematant pačios sistemos verdikto.

### 3 eksperimentiniai scenarijai

- **Scenarijus A** — pagrindinis srautas (5 agentai: profiliavimas → atitikimas → samprotavimas → refleksija → sprendimas)
- **Scenarijus B** — A + ESCO vaidmens kontekstas JD profiliavimo agentui
- **Scenarijus C** — B + dvipakopis kalibravimas su etikečių atminties banku (7 agentai)

### Refleksijos kilpa

Refleksijos agentas peržiūri samprotavimo agento išvestį dėl nuoseklumo (ar balas dera su trūkumais/stiprybėmis, ar stiprybės nesutampa su trūkumais ir t. t.). Aptikus prieštaravimą, orkestratorius grąžina samprotavimo agentą su grįžtamuoju ryšiu. Ciklas kartojamas iki `max_revisions` kartų (numatyta — 2).

## Diegimas

### 1. Virtualioji aplinka

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 2. Priklausomybės

```bash
pip install -r requirements.txt
```

### 3. Aplinkos kintamieji

```bash
cp .env.example .env
# Atidarykite .env ir įveskite savo API raktus
```

Reikalingi API raktai:
- `OPENAI_API_KEY` — `gpt-4o-mini` pagrindinei sistemai
- `JUDGE_API_KEY` — DeepSeek-V3.2 LLM-teisėjui (kitos šeimos modelis — esminis principas dėl šališkumo)
- `ANTHROPIC_API_KEY` — Claude Haiku *(tik augmentacijos eksperimentui)*

## Naudojimas

### Vienos CV-JD poros palyginimas

```bash
# Su pavyzdiniais failais ir Mock LLM (be API raktų)
python main.py

# Su realiu LLM
python main.py --real

# Konkretus scenarijus
python main.py --real --scenario A
python main.py --real --scenario B
python main.py --real --scenario C

# Su savo failais
python main.py --real --cv path/to/cv.txt --jd path/to/jd.txt
```

### Eksperimentas ant duomenų rinkinio

```bash
# Atsisiųsti Resume-Screening-Dataset iš HuggingFace
python data/load_hf_dataset.py --sample 5000

# Paleisti pagrindinį eksperimentą (visi 3 scenarijai + SBERT baseline)
python main.py --real --evaluate data/hf_test_5000.json --baseline

# TF-IDF + LR lyginamasis pagrindas (atskirai)
python scripts/run_tfidf_baselines.py
```

### LLM-teisėjo auditas

```bash
python audit.py --dataset data/hf_test_5000.json --output results/judge_audit.json
```

### Interaktyvus rezultatų dashboard'as

```bash
streamlit run dashboard.py
```

### Vienetiniai testai

```bash
python -m pytest tests/ -v
```

## Projekto struktūra

```
job-match/
├── agents/               # 8 specializuoti agentai + LLM-teisėjas
│   ├── cv_profiling.py
│   ├── jd_profiling.py
│   ├── matching.py
│   ├── reasoning.py
│   ├── reflection.py
│   ├── decision.py
│   ├── memory_retrieval.py
│   ├── calibration.py
│   └── judge.py          # Nepriklausomas LLM-teisėjas
├── orchestrator/         # Agentų grandinės koordinavimas + refleksijos kilpa
├── models/               # Pydantic duomenų modeliai (SharedContext ir kt.)
├── llm/                  # LLM klientų abstrakcija (OpenAI, DeepSeek, MockLLM)
├── embeddings/           # Sentence-BERT panašumas (all-MiniLM-L6-v2)
├── memory/               # Atminties banko logika
├── evaluation/           # Eksperimentų karkasas
│   ├── runner.py
│   ├── baseline.py       # SBERT lyginamasis pagrindas
│   └── metrics.py
├── config/               # Aplinkos kintamųjų valdymas (Settings klasė)
├── scripts/              # Pagalbiniai skriptai
│   ├── run_tfidf_baselines.py     # TF-IDF + LR baseline'as
│   ├── generate_thesis_plots.py   # Tezės paveikslų generavimas
│   └── compute_4_1_metrics.py     # Statistinės metrikos
├── tests/                # Vienetiniai testai
├── docs/                 # Papildoma dokumentacija
├── data/
│   ├── sample_cv.txt
│   ├── sample_jd.txt
│   └── load_hf_dataset.py         # Rinkinio atsisiuntimas iš HuggingFace
├── main.py               # CLI įėjimo taškas
├── audit.py              # LLM-teisėjo auditas
├── dashboard.py          # Streamlit dashboard'as
├── architecture.md       # Detali architektūros dokumentacija
├── requirements.txt
└── .env.example
```

## Technologijų rinkinys

- **Python 3.12**
- **pydantic 2.x** — struktūrinių duomenų modeliai
- **sentence-transformers** — semantiniai embedding'ai (`all-MiniLM-L6-v2`)
- **openai** — LLM API klientas (gpt-4o-mini; DeepSeek per OpenAI-suderinamą endpoint'ą)
- **anthropic** — Claude Haiku API (augmentacijos eksperimentui)
- **scikit-learn** — TF-IDF vektorizavimas, Logistic Regression, kosinusinis panašumas
- **streamlit** — interaktyvus dashboard'as
- **plotly, matplotlib** — vizualizacijos
- **pytest** — testavimo karkasas

## Akademinis kontekstas

Tai bakalauro darbo prototipas, sukurtas tyrimui apie agentinių LLM sistemų pritaikomumą CV ir darbo skelbimų semantinio atitikimo užduotyje. Pagrindiniai tyrimo klausimai:

- **TK1.** Ar daugiaagentė LLM-grįsta architektūra pranoksta klasikinius lyginamuosius pagrindus (TF-IDF + LR, Sentence-BERT)?
- **TK2.** Ar scenarijų progresija A→B→C duoda monotoniškai gerėjantį tikslumą?
- **TK3.** Kaip kinta sistemos vertinimas, kai vietoj šaltinio etikečių naudojamas nepriklausomo LLM-teisėjo verdiktas?
- **TK4.** Kaip architektūros nauda priklauso nuo pagrindinio LLM gebėjimo (`gpt-4o-mini` vs `Qwen 7B`)?

### Pagrindiniai radiniai

- ✅ **H1 patvirtinta** — agentinė architektūra pranoksta abu klasikinius baseline'us F1 atžvilgiu abiem atskaitos taškais
- ❌ **H2 paneigta** — scenarijų progresija A→B→C nedavė monotoniško pagerinimo
- 🔍 LLM-teisėjo auditas atskleidė **31,8 %** šaltinio etikečių klaidų ir **20,9 %** „šabloninių atmetimų" — pagrindinė H2 paneigimo priežastis
- 🔍 Tarp-modelinis sutapimas (gpt-4o-mini vs Qwen 7B) — tik **54,5 %**, parodantis stipriai modelio gebėjimo nulemtą sprendimų variabilumą
- 🔍 JD augmentacija (pratęsimas iki realistiško ilgio) F1 įvertį **sumažino**, ne padidino

## Atviras kodas

Repozitorija paskelbta viešai pagal atvirojo mokslo principą. Eksperimentų rezultatai, dideli duomenų rinkiniai ir lokali virtualioji aplinka į repozitoriją neįtraukti dėl jų apimties — viešąjį Resume-Screening-Dataset galima atsisiųsti su pridėtu `data/load_hf_dataset.py` skriptu.

## Autorius

**Karolis Miežetis**
Vilniaus universitetas, Matematikos ir informatikos fakultetas
Bakalauro darbas, 2026 m.
