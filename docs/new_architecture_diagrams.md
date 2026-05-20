# Naujos architektūros diagramos — Tier 2 perprojektavimas

Šis failas turi visas perprojektuotos sistemos diagramas: padalintus CV/JD profiliavimo agentus, pašalintą papildomos kontekstinės informacijos agentą, pridėtą dvipakopį sprendimą su atminties banko kalibravimu ir grynai online (per srautą) vykdomą vertinimą.

---

## Kaip žiūrėti šį failą kaip realias diagramas

Žemiau pateiktos diagramos parašytos Mermaid sintakse. Jos vizualiai atvaizduojamos:

- **VS Code** — įmontuotas Markdown peržiūros režimas (Ctrl+Shift+V) naujesnėse versijose atvaizduoja Mermaid. Jei jūsiškis neatvaizduoja, įdiekite plėtinį *„Markdown Preview Mermaid Support"* (autorius Matt Bierner).
- **GitHub** — įkėlus failą ir atvėrus github.com, Mermaid atvaizduojama natūraliai.
- **Obsidian, Typora, MarkText** — visi atvaizduoja Mermaid natūraliai.
- **Online** — bet kurį atskirą ` ```mermaid ... ``` ` bloką galima įklijuoti į <https://mermaid.live> greitam peržiūrai.

Jei matote kodo blokus vietoj diagramų, jūsų peržiūros įrankis Mermaid nepalaiko. Greičiausias sprendimas Windows aplinkoje — VS Code įmontuotas peržiūros režimas su minėtu plėtiniu.

---

## Kas keičiasi

| Komponentas | Statusas | Detalė |
|---|---|---|
| `ExtractionAgent` (viena klasė, vykdoma du kartus) | **PAŠALINTAS** | Pakeistas dviem specializuotais agentais |
| `CVProfilingAgent` | **NAUJAS** | Sukuria kandidato profilį iš CV |
| `JDProfilingAgent` | **NAUJAS** | Sukuria idealaus kandidato profilį iš JD su ESCO profesijos kontekstu |
| `ContextEnrichmentAgent` | **PAŠALINTAS** | Įgūdžių normalizavimas absorbuotas į abu profiliavimo agentus |
| `SemanticMatchingAgent` | nepakitęs | Vis dar skaičiuoja kosinusinį panašumą |
| `ReasoningAgent` | nepakitęs | Vis dar generuoja stiprybes/trūkumus/`suggested_score` |
| `ReflectionAgent` | vaidmuo nepakitęs | Tik 1-osios pakopos nuoseklumo patikrinimas — NEMATO etikečių |
| `DecisionAgent` | vaidmuo susiaurintas | Dabar generuoja tik **`initial_decision`** (1-oji pakopa) |
| `MemoryRetrievalAgent` | **ATNAUJINTAS** | Dabar grąžina `LabeledMemoryEntry` vietoj nepaženklintos atminties |
| `CalibrationAgent` | **NAUJAS** | 2-oji pakopa — peržiūri `initial_decision` palyginant su etikečių istorija, generuoja `final_decision` |
| Atminties įrašo formatas | **ATNAUJINTAS** | `MemoryEntry` → `LabeledMemoryEntry` su `ground_truth_label` + `ground_truth_reason` |
| Vertinimo protokolas | **NAUJAS** | Grynai online per srautą: pirma prognozuoja → prideda etiketę → išsaugo → kita pora |

---

## 1 diagrama: Scenarijus A (bazinis, be vaidmens konteksto, be atminties)

```mermaid
flowchart TD
    Start([CV + JD įvestis]) --> CVP[CVProfilingAgent<br/>sukuria kandidato profilį]
    Start --> JDP[JDProfilingAgent<br/>sukuria idealaus kandidato profilį<br/>BE ESCO konteksto]
    CVP --> Match[SemanticMatchingAgent<br/>kosinusinis panašumas]
    JDP --> Match
    Match --> Reason[ReasoningAgent<br/>stiprybės, trūkumai, suggested_score]
    Reason --> Refl{ReflectionAgent<br/>nuoseklumo patikrinimas}
    Refl -- reikia peržiūros<br/>≤ 2 ciklai --> Reason
    Refl -- nuoseklu --> Dec[DecisionAgent<br/>final_decision]
    Dec --> Final([Užfiksuotas sprendimas])
```

---

## 2 diagrama: Scenarijus B (A + ESCO vaidmens kontekstas JD profiliavime)

```mermaid
flowchart TD
    Start([CV + JD įvestis]) --> CVP[CVProfilingAgent]
    Start --> JDP[JDProfilingAgent<br/>SU ESCO vaidmens kontekstu]
    ESCO[(ESCO profesijos<br/>data/esco_occupations.json)] -. vaidmens profilio paieška .-> JDP
    CVP --> Match[SemanticMatchingAgent]
    JDP --> Match
    Match --> Reason[ReasoningAgent]
    Reason --> Refl{ReflectionAgent}
    Refl -- reikia peržiūros --> Reason
    Refl -- nuoseklu --> Dec[DecisionAgent<br/>final_decision]
    Dec --> Final([Užfiksuotas sprendimas])
```

---

## 3 diagrama: Scenarijus C — pilna detalė (naujasis srautas)

Tai labiausiai pakeistas scenarijus ir tezės palyginimo centras. Jis naudoja ESCO vaidmens kontekstą IR pažymėtą atmintį + dvipakopį kalibravimą.

```mermaid
flowchart TD
    Start([CV tekstas + JD tekstas]) --> CVP[CVProfilingAgent<br/>sukuria kandidato profilį<br/>įgūdžiai, vyresniškumas, archetipas]
    Start --> JDP[JDProfilingAgent<br/>sukuria idealaus kandidato profilį<br/>naudoja ESCO vaidmens kontekstą]
    ESCO[(ESCO<br/>profesijų failas)] -. vaidmens paieška .-> JDP

    CVP --> Match[SemanticMatchingAgent<br/>kosinusinio panašumo matrica]
    JDP --> Match

    Match --> Reason[ReasoningAgent<br/>vertina reikalavimus ATITIKTI/DALINAI/TRŪKSTAMA<br/>generuoja stiprybes, trūkumus, suggested_score]

    Reason --> Refl{ReflectionAgent<br/>nuoseklumas<br/>+ šališkumo patikrinimai<br/>NEMATO etikečių}
    Refl -- reikia peržiūros --> Reason
    Refl -- nuoseklu --> Dec1[DecisionAgent — 1 PAKOPA<br/>užfiksuoja initial_decision<br/>nematant jokių etikečių]

    Dec1 --> MR[MemoryRetrievalAgent<br/>grąžina top-K pažymėtų praeities porų<br/>pagal panašumą su dabartine]
    Mem[(Pažymėta atminties saugykla<br/>kiekvienas įrašas: profilis + system_score<br/>+ žmogaus etiketė + žmogaus priežastis)] --> MR

    MR --> Cal[CalibrationAgent — 2 PAKOPA<br/>lygina initial_decision su pažymėta istorija<br/>aptinka kalibravimo klaidas<br/>generuoja final_decision]

    Cal --> Final([final_decision<br/>užfiksuotas balas, pasitikėjimas,<br/>rekomendacija])

    Final -. prognozuoja pirma .-> Attach[Pridėti žmogaus etiketę<br/>+ pradinę priežastį iš rinkinio]
    Attach --> NewEntry[Sukurti LabeledMemoryEntry]
    NewEntry --> Mem
    Final --> NextPair([Kita pora prasideda čia])
```

### Pagrindinės C scenarijaus savybės

1. **Dvipakopis sprendimas**: 1 pakopa (Reasoning + Reflection + DecisionAgent) užfiksuoja `initial_decision` nematant jokių etikečių. 2 pakopa (MemoryRetrieval + CalibrationAgent) peržiūri `initial_decision` palyginant su pažymėta istorija ir užfiksuoja `final_decision`.
2. **Etikečių nutekėjimo prevencija**: 1 pakopoje sistema nemato nieko apie tikrąją etiketę. Etiketė pridedama tik PO TO, kai `final_decision` jau užfiksuotas.
3. **Cold-start (grynai online)**: atmintis pradžioje tuščia. 1-oji pora turi nulį pažymėtos istorijos; 150-oji — 149 įrašus.
4. **Refleksija ir kalibravimas — skirtingi mechanizmai**: Refleksija tikrina vidinį nuoseklumą (be etikečių). Kalibravimas naudoja išorinį pažymėtą grįžtamąjį ryšį (su etiketėmis). Švarus atskyrimas.

---

## 4 diagrama: Srauto vertinimo protokolas laike

```mermaid
sequenceDiagram
    participant R as ExperimentRunner
    participant Agents as Agentų grandinė<br/>(1 + 2 pakopa)
    participant M as Pažymėta atminties saugykla
    participant D as Duomenų rinkinys

    Note over M: pradžioje tuščia (cold start)
    Note over R: 1-OJI PORA
    R->>D: imti 1-ąją porą
    D-->>R: cv, jd, etiketė, priežastis
    R->>Agents: prognozuoti (etiketė PASLĖPTA)
    Agents->>M: imti top-K
    M-->>Agents: [] tuščia atmintis
    Note right of Agents: tik 1 pakopa<br/>(kalibravimas neįmanomas)
    Agents-->>R: final_decision = initial_decision
    R->>R: palyginti su etikete, registruoti metrikas
    R->>M: išsaugoti LabeledMemoryEntry<br/>(sprendimas + etiketė + priežastis)

    Note over R: 2-OJI PORA
    R->>D: imti 2-ąją porą
    D-->>R: cv, jd, etiketė, priežastis
    R->>Agents: prognozuoti (etiketė PASLĖPTA)
    Agents->>M: imti top-K
    M-->>Agents: [1-osios poros pažymėtas įrašas]
    Note right of Agents: 1 pavyzdys kalibravimui<br/>2 pakopa gali pradėti padėti
    Agents-->>R: final_decision
    R->>M: išsaugoti pažymėtą įrašą

    Note over R: 150-OJI PORA
    R->>D: imti 150-ąją porą
    R->>Agents: prognozuoti (etiketė PASLĖPTA)
    Agents->>M: imti top-K iš 149 įrašų
    M-->>Agents: [k panašiausi pažymėti praeities atvejai]
    Note right of Agents: turtinga istorija<br/>stiprus kalibravimo signalas
    Agents-->>R: final_decision
    R->>M: išsaugoti pažymėtą įrašą
```

Tai yra **StreamBench protokolas** (Yehudai et al. 2025 §3 — nuolatinis tobulinimas iš srauto duomenų su grįžtamuoju ryšiu). Sistema niekada nemokoma pažymėtais duomenimis — kiekviena prognozė užfiksuojama prieš atskleidžiant jos etiketę. Praeities porų etiketės tampa kontekstine medžiaga tolesnėms poroms.

---

## 5 diagrama: `LabeledMemoryEntry` struktūra

```mermaid
classDiagram
    class LabeledMemoryEntry {
        +str memory_id
        +str timestamp
        +str cv_profile_summary
        +str jd_profile_summary
        +float system_score
        +str system_recommendation
        +str system_reasoning_summary
        +bool ground_truth_label
        +str ground_truth_reason
        +bool was_correct
        +str error_direction
        +list~str~ influenced_by
        +float similarity_to_current
    }

    note for LabeledMemoryEntry "system_score sukuriamas PRIEŠ matant etiketę.\nground_truth_label ir ground_truth_reason pridedami PO prognozės.\nwas_correct ir error_direction išvedami kai abu žinomi."
```

### Laukų semantika

| Laukas | Kada nustatomas | Šaltinis |
|---|---|---|
| `memory_id` | Sukūrimo metu | UUID, generuojamas runner'io |
| `cv_profile_summary` | Sukūrimo metu | Iš `CandidateProfile.raw_summary` |
| `jd_profile_summary` | Sukūrimo metu | Iš `IdealCandidateProfile.raw_summary` |
| `system_score` | Po 2 pakopos užfiksuoto `final_decision` | Iš `final_decision.score` |
| `system_recommendation` | Po 2 pakopos | Iš `final_decision.recommendation` |
| `system_reasoning_summary` | Po 2 pakopos | Iš `reasoning_output.overall_assessment` |
| `ground_truth_label` | Po prognozės užfiksavimo | Iš rinkinio (`Decision` laukas) |
| `ground_truth_reason` | Po prognozės | Iš rinkinio (`Reason_for_decision` laukas) |
| `was_correct` | Išvedamas | `(system_score >= riba) == ground_truth_label` |
| `error_direction` | Išvedamas | Vienas iš {TP, FP, TN, FN} |
| `influenced_by` | Po 2 pakopos | Sąrašas `memory_id`, paimtų šios poros 2 pakopos metu |
| `similarity_to_current` | Paėmimo metu | Nustatomas, kai šis įrašas paimamas tolesnei porai |

---

## 6 diagrama: Kas pašalinta, palikta ir pridėta

```mermaid
flowchart TD
    subgraph OLD["Sena architektūra"]
        O1[ExtractionAgent<br/>kviečiamas du kartus CV ir JD]
        O2[ContextEnrichmentAgent<br/>ESCO įgūdžių normalizavimas]
        O3[MemoryRetrievalAgent<br/>nepaženklinta atmintis]
        O4[ReflectionAgent<br/>nuoseklumo patikrinimas]
        O5[DecisionAgent<br/>viena pakopa]
    end

    subgraph NEW["Nauja architektūra (Tier 2)"]
        N1A[CVProfilingAgent<br/>NAUJAS]
        N1B[JDProfilingAgent<br/>NAUJAS. Naudoja ESCO profesijas]
        N3[MemoryRetrievalAgent<br/>ATNAUJINTAS LabeledMemoryEntry]
        N4[ReflectionAgent<br/>NEPAKITĘS. Tik 1 pakopa<br/>nuoseklumas, BE etikečių]
        N5[DecisionAgent<br/>SUSIAURINTAS. Užfiksuoja initial_decision<br/>tik 1 pakopa]
        N6[CalibrationAgent<br/>NAUJAS. 2 pakopa.<br/>Užfiksuoja final_decision<br/>naudojant pažymėtą istoriją]
    end

    O1 -. padalintas į .-> N1A
    O1 -. padalintas į .-> N1B
    O2 -. absorbuotas į .-> N1B
    O3 -. atnaujintas .-> N3
    O4 -. paliktas .-> N4
    O5 -. tampa tik 1 pakopa .-> N5
```

---

## 7 diagrama: 1 pakopa vs 2 pakopa — atskyrimas

```mermaid
flowchart LR
    subgraph P1["1 PAKOPA — nepriklausomas sprendimas (be etikečių)"]
        direction LR
        P1A[CVProfiling] --> P1C[Matching]
        P1B[JDProfiling] --> P1C
        P1C --> P1D[Reasoning]
        P1D --> P1E{Reflection}
        P1E -- nuoseklu --> P1F[DecisionAgent]
        P1E -- peržiūrėti --> P1D
        P1F --> P1OUT[initial_decision]
    end

    subgraph P2["2 PAKOPA — pažymėtos atminties kalibravimas"]
        direction LR
        P2A[MemoryRetrievalAgent]
        P2B[CalibrationAgent]
        P2A -- pažymėtos praeities poros --> P2B
        P2B --> P2OUT[final_decision]
    end

    P1OUT --> P2A
```

### Kodėl dvi pakopos (o ne vienas didelis agentas, kuris mato viską)

- **Švaresnė ablacija**: galime pranešti `initial_decision` IR `final_decision` kiekvienai porai. Delta = tiesioginis matavimas, ar pažymėta atmintis padėjo šioje poroje.
- **Švaresnis priskyrimas**: 1 pakopos rezultatai tiesiogiai palyginami tarp visų trijų scenarijų. Scenarijaus A 1 pakopa == galutinis A. Scenarijaus C 1 pakopa == „ką C būtų padaręs be atminties". Tai izoliuoja atminties indėlį.
- **Švaresnė tezės istorija**: „agentas pirma sprendžia pats, paskui apmąsto pažymėtą istoriją ir peržiūri." Atspindi žmogaus pažinimą.
- **Švaresnė metodologija**: „vidinio samprotavimo" (1 pakopa) atskyrimas nuo „išorinio grįžtamojo ryšio" (2 pakopa) yra pripažintas agentų dizaino šablonas literatūroje.

---

## 8 diagrama: Visi trys scenarijai vienoje drobėje (aukšto lygio palyginimas)

```mermaid
flowchart TB
    subgraph A["SCENARIJUS A"]
        A1[CVProfiling] --> A3[Matching]
        A2[JDProfiling be ESCO] --> A3
        A3 --> A4[Reasoning + Reflection kilpa]
        A4 --> A5[DecisionAgent]
        A5 --> A6([final])
    end

    subgraph B["SCENARIJUS B"]
        B1[CVProfiling] --> B3[Matching]
        B2[JDProfiling + ESCO] --> B3
        ESCOB[(ESCO)] -. vaidmuo .-> B2
        B3 --> B4[Reasoning + Reflection kilpa]
        B4 --> B5[DecisionAgent]
        B5 --> B6([final])
    end

    subgraph C["SCENARIJUS C"]
        C1[CVProfiling] --> C3[Matching]
        C2[JDProfiling + ESCO] --> C3
        ESCOC[(ESCO)] -. vaidmuo .-> C2
        C3 --> C4[Reasoning + Reflection kilpa]
        C4 --> C5[DecisionAgent — 1 pakopa]
        C5 --> C6[MemoryRetrieval]
        MemC[(Pažymėta atmintis)] --> C6
        C6 --> C7[CalibrationAgent — 2 pakopa]
        C7 --> C8([final])
        C8 -. pridėti etiketę, išsaugoti .-> MemC
    end
```

Tai daro eksperimentinį kontrastą aiškų:

- **A vs B** matuoja ESCO vaidmens konteksto indėlį.
- **B vs C 1 pakopa** yra niekinis veiksmas (1 pakopa identiška tarp B ir C).
- **C 1 pakopa vs C 2 pakopa** matuoja pažymėtos atminties kalibravimo indėlį.
- **A vs C final** matuoja bendrą vaidmens konteksto + pažymėtos atminties indėlį.

---

## Patvirtinti dizaino sprendimai

| Sprendimas | Pasirinkimas |
|---|---|
| Padalinti ekstrakciją į CVProfiling + JDProfiling | ✅ patvirtinta |
| Pašalinti `ContextEnrichmentAgent` | ✅ patvirtinta (darbas absorbuotas į JDProfilingAgent) |
| Dvipakopis sprendimas (initial + kalibruotas) | ✅ patvirtinta |
| Atminties formatas: saugoti sprendimą + tikrąją etiketę + priežastį | ✅ patvirtinta (LabeledMemoryEntry) |
| Srauto protokolas: pirma prognozuoti, paskui pridėti etiketę ir išsaugoti | ✅ patvirtinta |
| Cold-start atmintis (tuščia vertinimo pradžioje, grynai online) | ✅ patvirtinta |
