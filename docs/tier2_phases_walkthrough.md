# Tier 2 architektūra — 1–4 fazių apžvalginis vadovas

Šis dokumentas paaiškina, kas buvo sukurta per Tier 2 perprojektavimo 1–4 fazes — failas po failo, su pakeitimų motyvacija.

---

## Bendras vaizdas: kas pasikeitė

Originalioje (Tier 1) architektūroje buvo viena `ExtractionAgent` klasė, vykdoma du kartus — kartą ant CV, kartą ant JD — po kurios sekė `ContextEnrichmentAgent`, atliekantis ESCO įgūdžių normalizavimą. Atmintis scenarijuje C saugojo sistemos pačios praeities sprendimus (be tikrųjų etikečių), o tai pasirodė esąs šališkas „aido kambarys".

Naujoji (Tier 2) architektūra ekstrakcijos sluoksnį pakeičia dviem specializuotais profiliavimo agentais, kurie generuoja turtingesnius, interpretacinius profilius. Atmintis dabar saugo sistemos sprendimus PLUS žmogaus tikrąsias etiketes, įgalindama **dvipakopio sprendimo srautą** scenarijuje C: pradinis sprendimas (1 pakopa), po kurio seka pažymėtos atminties kalibravimas (2 pakopa).

Abi architektūros koegzistuoja kodų bazėje. Numatytasis yra Tier 1 (paveldas išsaugotas). Naudokite `--architecture tier2`, kad pasirinktumėte naują.

---

## 1 fazė — Pamatas (duomenų modeliai + atmintis + srauto protokolas)

**Tikslas**: apibrėžti naujas duomenų sutartis ir atminties infrastruktūrą, kuriomis remiasi 2–4 fazės. Jokių agentų pakeitimų dar.

### `models/entities.py` — keturi nauji Pydantic modeliai

| Modelis | Paskirtis |
|---|---|
| `CandidateProfile` | Ką generuos `CVProfilingAgent`. Plečia žalius įgūdžius/patirtį/išsilavinimą interpretaciniais laukais: `seniority_level`, `domain_expertise`, `candidate_archetype`, `likely_role_fit`. |
| `IdealCandidateProfile` | Ką generuos `JDProfilingAgent`. Turi `required_skills` (JD eksplicitiški) ir `typical_role_skills` (ESCO išvesti) atskirai. Apima `detected_role`, `role_confidence`, `esco_code`. |
| `CalibrationOutput` | Ką generuos `CalibrationAgent` (2 pakopa). Laukai: `calibration_decision` ('lower'/'raise'/'keep'), `adjusted_score`, `rationale`, `pattern_observed`, `n_supporting_memories`. |
| `LabeledMemoryEntry` | Naujas atminties įrašas. Saugo `system_score` IR `ground_truth_label` IR `ground_truth_reason`. Apima `pair_id` persidengimo aptikimui, `was_correct` ir `error_direction` (TP/FP/TN/FN) kaip išvedinius laukus. |

Originalūs `MemoryEntry`, `ExtractedEntities` ir `NormalizedEntities` paliekami nepaliesti, kad Tier 1 vis dar veiktų.

### `models/shared_context.py` — nauji laukai lentoje

Pridėti (kartu su esamais laukais):
- `cv_profile: Optional[CandidateProfile]`
- `jd_profile: Optional[IdealCandidateProfile]`
- `initial_decision: Optional[FinalDecision]` — 1 pakopos užfiksavimas
- `calibration_output: Optional[CalibrationOutput]` — 2 pakopos audito pėdsakas
- `labeled_memory_entries: list[LabeledMemoryEntry]` — Tier 2 paieškos rezultatai
- Nauji pagalbiniai metodai: `has_profiles()`, `has_labeled_memory()`

### `memory/labeled_store.py` — naujas failas, atspindintis `MemoryStore`, bet pažymėtiems įrašams

`LabeledMemoryStore` turi tas pačias savybes kaip ir paveldima saugykla (kosinusinis paėmimas, šviežumo svoris, dedubliavimas prie 0,95 ribos), bet:
- Prideda `pair_ids() -> set[str]` metodą persidengimo aptikimui
- Saugo `pair_id`, kad runner'is galėtų aptikti pakartotinio vertinimo bandymus
- Išsaugo į `labeled_memories.json` + `labeled_embeddings.npy` (atskirti failai nuo paveldimų — sena ir nauja atmintis gali koegzistuoti diske)

### `evaluation/streaming_protocol.py` — naujas failas, srauto vertinimo logika

Dvi funkcijos:

- **`prepare_memory_store(memory_dir, mode, input_pair_ids)`** — inicializuoja pažymėtą atminties saugyklą pagal vieną iš trijų režimų:
  - `cold-start`: išvalyti atminties aplanką, pradėti tuščią. Numatytasis tezės vykdymams.
  - `continue-stream`: išsaugoti esamą atmintį; **mesti `StreamingProtocolError`, jei kuris nors įvesties pair_id jau yra atmintyje** (persidengimo aptikimas užkerta kelią testavimo rinkinio nutekėjimui).
  - `fresh-build`: tas pats efektas kaip cold-start, bet įspėja, jei atmintis egzistavo.
- **`build_labeled_entry(...)`** — sukuria `LabeledMemoryEntry` po to, kai prognozė užfiksuota, automatiškai išvesdamas `was_correct` ir `error_direction` (TP/FP/TN/FN) iš ribos.

Čia įgyvendinamas StreamBench protokolo invariantas: poros, kurios jau prisidėjo etiketėmis prie atminties, negali būti pakartotinai vertinamos.

### Patikrinimas

Visas 1 fazės kodas buvo dūmų testas su mock LLM. Paveldima Tier 1 grandinė po to vykdėsi nepakitusi.

---

## 2 fazė — Profiliavimo agentai

**Tikslas**: pakeisti vieną `ExtractionAgent` (vykdomą du kartus) dviem specializuotais agentais, kurie generuoja turtingesnius profilius.

### `data/esco_occupations.json` — pradinė versija (12 įrašų; 3 fazėje išplėsta iki 105)

Naujas duomenų failas, susiejantis vaidmens raktus su ESCO profesijų profiliais. Kiekvienas įrašas turi:
- `esco_code`
- `preferred_label` ir `alt_labels`
- `description`
- `typical_skills`
- `typical_experience_years`
- `typical_education`
- `typical_responsibilities`
- `seniority_levels`

`JDProfilingAgent` šį failą perskaito konstravimo metu ir naudoja kaip vaidmens klasifikavimo atskaitą.

### `agents/cv_profiling.py` — naujas

`CVProfilingAgent` sukuria `CandidateProfile` iš CV. Promptas instruktuoja LLM generuoti tiek žalią įgūdžių ekstrakciją, tiek interpretacinius laukus, tokius kaip `candidate_archetype` (vieno sakinio santrauka) ir `likely_role_fit` (kurį ESCO vaidmenį šis CV labiausiai primena). Išvestis validuojama pagal `CandidateProfile` Pydantic modelį.

### `agents/jd_profiling.py` — naujas

`JDProfilingAgent` vykdomas **dvejomis pakopomis**:

1. **Vaidmens klasifikacija** (1 pakopa): mažas LLM iškvietimas, gaunamas JD tekstas ir kuruoto vaidmenų vardų sąrašo iš ESCO failo. Grąžina `detected_role` ir `role_confidence`. Jei pasitikėjimas < 0,5, agentas grįžta į bendrinį profilį.
2. **Profiliavimas** (2 pakopa): pagrindinis LLM iškvietimas. Gauna JD tekstą PLIUS ESCO kontekstą aptiktam vaidmeniui (typical_skills, typical_experience, typical_responsibilities). Grąžina `IdealCandidateProfile`.

Agentas turi `use_esco_context` konstruktoriaus vėliavą — Scenarijus A perduoda `False` (be ESCO), Scenarijai B/C perduoda `True`. Haliucinacijų apsauga: jei LLM grąžina vaidmens vardą, kurio nėra ESCO faile, agentas grįžta į bendrinį profilį.

### `llm/client.py` — `MockLLMClient` praplėstas

Pridėtas prompto šablono aptikimas naujiems agentams (`cv profiling agent`, `role classification agent`, `jd profiling agent`, `calibration agent`). Kiekvienas grąžina kanoninį mock JSON, atitinkantį agento Pydantic išvesties modelį. Leidžia visus 2–4 fazės darbus testuoti dūmų testais be API iškvietimų.

---

## 3 fazė — Kuruoti ESCO profesijų duomenys

**Tikslas**: išplėsti `data/esco_occupations.json` nuo 12 įrašų pradinės versijos iki ginama 105 įrašų kuruoto rinkinio.

### Apimties strategija

| Kategorija | Skaičius |
|---|---|
| Visi unikalūs vaidmenys vertinimo rinkinyje | 38/38 padengta |
| Tech/data (inžinerija, ML, duomenys, sauga ir kt.) | 43 |
| Sveikatos apsauga | 8 |
| Finansai | 6 |
| Pardavimai/Rinkodara | 8 |
| Operacijos | 5 |
| Amatai | 5 |
| Kūryba/Žiniasklaida | 6 |
| Apgyvendinimas/Mažmena | 5 |
| Švietimas | 4 |
| Mokslas | 4 |
| Kita | likę |

Visų rinkinio vaidmenų įtraukimas užtikrina, kad kiekviena testavimo pora randa savo specifinį vaidmens profilį. ~60+ ne-tech vaidmenų įtraukimas demonstruoja, kad sistema nėra perdirbusi testavimo paskirstymui — ji gali apdoroti JD iš sričių, kurioms ji nebuvo derinta.

### Kainos pastaba

Išplėstas vaidmenų sąrašas prideda ~5K žetonų prie 1 pakopos prompto (~$0.0007 porai ant `gpt-4o-mini`). Niekinis 150 porų mastu.

---

## 4 fazė — Kalibravimo agentas + orkestratoriaus integracija

**Tikslas**: sukurti 2 pakopą (kalibravimą), sujungti visus naujus agentus per orkestratorių ir atnaujinti runner'į, kad jis taikytų srauto protokolą.

### `agents/calibration.py` — naujas (Tier 2 centras)

`CalibrationAgent` yra dvipakopio sprendimo srauto 2 pakopa:

1. Skaito `context.initial_decision` (1 pakopos užfiksavimą)
2. Skaito `context.labeled_memory_entries` (pažymėtas praeities poras, kurias paėmė `LabeledMemoryRetrievalAgent`)
3. Kviečia LLM su promptu, kuris sako: *„Štai sistemos pradinis sprendimas dabartinei porai. Štai panašios praeities poros SU tikrosiomis etiketėmis. Aptikite bet kokį miskalibravimo šabloną (per aukšti balai, per žemi balai) ir pasiūlykite koregavimą."*
4. Rašo `context.calibration_output` (pilną audito pėdsaką) ir `context.final_decision` (kalibruotą balą)

**Cold-start kelias**: kai `context.labeled_memory_entries` tuščias (tipiškai pirmajai cold-start vykdymo porai), `CalibrationAgent` visiškai praleidžia LLM iškvietimą ir tiesiog kopijuoja `initial_decision` į `final_decision`. Sutaupo žetonų iškvietimą, kai nėra prieš ką kalibruoti.

Agentas niekada nemato dabartinės poros savo etiketės — tik praeities porų etiketes. Tai StreamBench invariantas.

### `agents/labeled_memory_retrieval.py` — naujas

Tier 2 pakaitalas paveldimam `MemoryRetrievalAgent`. Skaito iš `LabeledMemoryStore` vietoj nepažymėto `MemoryStore`. Naudoja profilius paieškos užklausai sukurti, kai jie prieinami (turtingesnis signalas nei žalias tekstas). Jokio LLM iškvietimo — grynai embedding'ais grindžiamas paėmimas.

### Esami agentai — praplėsti profilių žinojimui

Trys esami agentai gavo nedidelius atnaujinimus, kad veiktų tiek su Tier 1 entitetais, tiek su Tier 2 profiliais:

- **`agents/reasoning.py`** — `_build_prompt()` dabar pirmenybę teikia `cv_profile`/`jd_profile`, kai jie yra. Tier 2 prompto sekcijos yra turtingesnės: kandidato archetipas, vyresniškumas, aptiktas vaidmuo, ESCO tipiški įgūdžiai eksplicitiškai atskirti nuo JD reikalaujamų įgūdžių.
- **`agents/decision.py`** — `_build_prompt()` skaito profilius, kai jie yra, IR `__init__` gavo `commits_to: str = "final"` parametrą. Kai `commits_to="initial"`, agentas rašo į `context.initial_decision` vietoj `context.final_decision` — naudojama Tier 2 scenarijaus C 1 pakopoje.
- **`models/shared_context.py`** — `get_skills_for_matching()` dabar pirmenybę teikia profiliams virš papildytų/žalių entitetų. JD profilių atveju grąžina dedubluotą `required_skills + typical_role_skills` sąjungą, kad matcher matytų pilną įgūdžių rinkinį, kurio reikalauja vaidmuo.

### `orchestrator/orchestrator.py` — gauna `architecture` parametrą ir Tier 2 grandinę

`Orchestrator.__init__` dabar priima `architecture: str = "tier1"`. Kai `"tier2"`, naujas `_build_tier2_chain()` metodas konstruoja:

- **Scenarijus A**: `CVProfilingAgent + JDProfilingAgent(no ESCO) + Matching + Reasoning + Reflection + DecisionAgent(commits_to=final)`
- **Scenarijus B**: tas pats, bet `JDProfilingAgent(use_esco_context=True)`
- **Scenarijus C**: tas pats kaip B + `LabeledMemoryRetrievalAgent + CalibrationAgent` po `DecisionAgent(commits_to=initial)` — pilnas dvipakopis srautas

ESCO yra vienintelis skirtumas tarp scenarijaus A ir B Tier 2. Atmintis + kalibravimas yra vienintelis skirtumas tarp B ir C. Kiekvienas scenarijus izoliuoja vieną architektūrinį kintamąjį.

### `evaluation/runner.py` — gauna srauto protokolą Tier 2

`ExperimentConfig` gavo du laukus: `architecture` ir `streaming_memory_mode`. `ExperimentRunner.run_all()`:

1. Jei `architecture="tier2"` ir scenarijus C yra vykdyme, kviečia `prepare_memory_store(...)` iš `streaming_protocol.py`, kad nustatytų pažymėtą atminties saugyklą pagal pasirinktą režimą.
2. `_run_single()` perduoda pažymėtą atminties saugyklą į Orchestrator.
3. **Po** kiekvienos poros `final_decision` užfiksavimo, `_run_single()` sukuria `LabeledMemoryEntry` (per `build_labeled_entry()`), prideda tikrąją etiketę ir priežastį iš rinkinio ir išsaugo į saugyklą. Kita pora galės paimti šį įrašą per 2 pakopą.

`CVJDPair` gavo `reference_reason` lauką, užpildytą `load_dataset_from_json` iš rinkinio `Reason_for_decision` stulpelio. Tai prijungiama prie kiekvieno `LabeledMemoryEntry`.

### `main.py` — naujos CLI vėliavos

- `--architecture tier1|tier2` (numatyta `tier1`)
- `--streaming-memory-mode cold-start|continue-stream|fresh-build` (numatyta `cold-start`)

---

## Dashboard'o atnaujinimai — Tier 2 analizės skirtukas

Naujas skirtukas „Tier 2 Analysis" `dashboard.py` faile. Reikšmingas tik vykdymams su `architecture=tier2`; Tier 1 vykdymams rodo informacinę žinutę.

Skirtukas turi keturias sekcijas:

1. **Two-Pass Decision Impact** — KPI kortelės, rodančios kiek scenarijaus C porų buvo sumažintos/padidintos/paliktos per kalibravimą; sprendimų pyrago diagrama; koregavimo dydžių histograma; 1 pakopa vs 2 pakopa sklaida (įstrižainė = jokio kalibravimo poveikio); KPI lyginantys 1 pakopos tikslumą su 2 pakopos tikslumu ir skaičiuojantys apsivertimus link/nuo tikrosios etiketės. *Tai pagrindinė vieta — tiesiogiai atsako į „ar pažymėtos atminties kalibravimas padėjo?"*
2. **ESCO Role Classification** — aptiktų vaidmenų pasiskirstymas, vaidmens pasitikėjimo histograma, atsarginių (`generic_professional`) atvejų skaičius. Rodo, ar ESCO klasifikacija veikia.
3. **CV ↔ JD Profile Match** — sutapimo dažnis tarp kandidato `likely_role_fit` ir JD `detected_role`. Aukštas sutapimas = sistema „mato" prasmingą atitikimą.
4. **Seniority Comparison** — porų skaičius, kuriose kandidato vyresniškumas atitinka JD reikalavimą vs nepakankamai kvalifikuotas vs per daug kvalifikuotas.

Esami dashboard'o skirtukai (Overview, Classification Metrics, Threshold Sensitivity ir kt.) veikia nepakitę ant Tier 2 rezultatų — rezultatų failo JSON schema nepasilaužė, tik buvo praplėsta.

---

## Kaip vykdyti Tier 2 vertinimą

```powershell
# 20-poros Tier 2 pilotas, visi scenarijai, cold-start atmintis
python main.py --real --evaluate data/hf_test_20.json --baseline `
    --architecture tier2 --threshold 60

# 150-poros Tier 2 vertinimas, visi scenarijai, persistuojantis pažymėtos atminties aplankas
python main.py --real --evaluate data/hf_test_150.json --baseline `
    --architecture tier2 --threshold 60 `
    --memory-dir data/memory_tier2_v1

# Continuing-stream praplėtimas (po vykdymo ant kito rinkinio pirma)
python main.py --real --evaluate data/hf_test_extra_150.json --baseline `
    --architecture tier2 --threshold 60 `
    --memory-dir data/memory_tier2_v1 `
    --streaming-memory-mode continue-stream

# Tier 1 (paveldima, numatyta — eksplicitiškai aiškumui)
python main.py --real --evaluate data/hf_test_150.json --baseline `
    --architecture tier1 --threshold 60
```

Po vykdymo, `streamlit run dashboard.py` ir pasirinkite rezultatų failą. Tier 2 Analysis skirtukas užsipildys. Run Comparison skirtukas gali palyginti Tier 1 vs Tier 2 rezultatus iš to paties rinkinio.

---

## Kas buvo sąmoningai išsaugota

| Dalykas | Kodėl jis vis dar yra |
|---|---|
| Tier 1 `ExtractionAgent`, `ContextEnrichmentAgent` | Atgalinis suderinamumas. `--architecture tier1` (numatytas) vis dar veikia. |
| Paveldimas `MemoryEntry` ir `MemoryStore` | Vis dar naudojamas Tier 1 scenarijaus C. Naujas kodas naudoja `LabeledMemoryEntry` ir `LabeledMemoryStore` atskirai. |
| Visi esami dashboard'o skirtukai | Rezultatų failo JSON schema yra griežtas viršrinkinis — seni skirtukai vis dar atvaizduoja Tier 2 rezultatus teisingai. |
| Refleksijos-peržiūros kilpa | Tas pats nuoseklumo mechanizmas abiejose architektūrose. 1 pakopa Tier 2 vis dar naudoja `ReflectionAgent`; kalibravimas yra atskiras 2 pakopos sluoksnis. |
| Rinkinio failai (`hf_test_150.json` ir kt.) | Nepakitę. Abi architektūros vertinamos ant tų pačių duomenų. |

---

## Kas dar NEatlikta

**5 fazė** būtų faktiniai vertinimo vykdymai:
- 20-poros Tier 2 pilotas (~$0.20, ~25 min) → patvirtinti, kad srautas veikia nuo pradžios iki pabaigos ant tikro LLM
- 150-poros Tier 2 vertinimas (~$1.50, ~3–4 valandos) → generuoti tezės kokybės skaičius
- Tiesioginis Tier 1 vs Tier 2 palyginimas dashboard'o Run Comparison skirtuke

Šie dar nevykdyti — architektūra paruošta, LLM kaina bus patirta, kai nuspręsite vykdyti tikrą vertinimą.

---

## Failas po failo — naujų + modifikuotų suvestinė

| Failas | Statusas |
|---|---|
| `models/entities.py` | Modifikuotas — pridėti `CandidateProfile`, `IdealCandidateProfile`, `LabeledMemoryEntry`, `CalibrationOutput` |
| `models/shared_context.py` | Modifikuotas — pridėti `cv_profile`, `jd_profile`, `initial_decision`, `calibration_output`, `labeled_memory_entries` laukai; atnaujintas `get_skills_for_matching()` |
| `memory/labeled_store.py` | **Naujas** — `LabeledMemoryStore` su `pair_ids()` persidengimo aptikimu |
| `evaluation/streaming_protocol.py` | **Naujas** — `prepare_memory_store()` ir `build_labeled_entry()` |
| `data/esco_occupations.json` | **Naujas** — 105 kuruotos ESCO profesijos per 12 sričių |
| `agents/cv_profiling.py` | **Naujas** — `CVProfilingAgent` |
| `agents/jd_profiling.py` | **Naujas** — `JDProfilingAgent` su dviem pakopomis: vaidmens klasifikacija + profiliavimas |
| `agents/calibration.py` | **Naujas** — `CalibrationAgent` (2 pakopa) |
| `agents/labeled_memory_retrieval.py` | **Naujas** — `LabeledMemoryRetrievalAgent` |
| `agents/reasoning.py` | Modifikuotas — `_build_prompt()` skaito profilius, kai jie yra |
| `agents/decision.py` | Modifikuotas — `commits_to` konstruktoriaus parametras; `_build_prompt()` skaito profilius |
| `llm/client.py` | Modifikuotas — `MockLLMClient` atpažįsta naujų agentų promptus |
| `orchestrator/orchestrator.py` | Modifikuotas — `architecture` parametras; naujas `_build_tier2_chain()` |
| `evaluation/runner.py` | Modifikuotas — `ExperimentConfig.architecture`, srauto protokolo iškvietimas, etiketės prijungimas po prognozės; praplėstas `TraceRecord` su Tier 2 laukais; `architecture` išsaugotame JSON konfigūracijoje |
| `main.py` | Modifikuotas — `--architecture` ir `--streaming-memory-mode` CLI vėliavos |
| `dashboard.py` | Modifikuotas — naujas „Tier 2 Analysis" skirtukas su 4 sub-sekcijomis |
| `docs/new_architecture_diagrams.md` | (Ankstesnis) Architektūros diagramos |
| `docs/tier2_phases_walkthrough.md` | (Šis failas) Apžvalginis vadovas |

Grynas kodo papildymas: ~1 200 eilučių per 5 naujus failus + ~400 eilučių modifikacijų. Joks esamas funkcionalumas nepašalintas.
