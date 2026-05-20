# Architektūra: CV ir darbo skelbimų semantinio atitikimo agentinė sistema

## 1. Architektūrinis šablonas: lentos (blackboard) paradigma

Ši sistema remiasi **lentos (angl. blackboard) paradigma** — gerai įsitvirtinusiu daugiaagentės dirbtinio intelekto šablonu (Erman et al., 1980; Corkill, 1991).

### Kodėl lenta, o ne grandinė (pipeline)?

**Grandinė** perduoda duomenis tiesiškai: žingsnis A → žingsnis B → žingsnis C. Kiekvienas žingsnis mato tik savo pirmtako išvestį.

**Lenta** yra fundamentaliai skirtinga:

```
┌─────────────────────────────────────────────────┐
│           BENDROS BŪSENOS KONTEKSTAS            │
│              (lenta / būsena)                    │
│                                                 │
│  cv_text, jd_text, cv_profile, jd_profile,      │
│  similarity_scores, reasoning_output,           │
│  reflection_output, revision_needed,            │
│  memory_matches, calibration, final_decision    │
└──┬──────┬──────┬──────┬──────┬──────┬──────┬───┘
   │      │      │      │      │      │      │
   ▼      ▼      ▼      ▼      ▼      ▼      ▼
┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌──────────┐
│ CV ││ JD ││Atit││Sam-││Refl││Spr-││Atm.+Kalib│
│prof││prof││ik. ││prot││eks.││en. ││ (scen. C)│
└────┘└────┘└────┘└────┘└────┘└────┘└──────────┘
```

Pagrindiniai skirtumai nuo grandinės:

1. **Pilnas būsenos matomumas**: kiekvienas agentas gali skaityti VISĄ bendros būsenos kontekstą, ne tik pirmtako išvestį. Samprotavimo agentas mato žalius tekstus, profilius IR panašumo balus vienu metu.

2. **Sąlyginis vykdymas**: orkestratorius sprendžia, kurie agentai vykdomi priklausomai nuo dabartinės būsenos. Scenarijuje A ESCO konteksto papildymas praleidžiamas. Scenarijuje C papildomai įjungiami atminties paieškos ir kalibravimo agentai.

3. **Iteratyvus tobulinimas**: refleksijos agentas gali nustatyti `needs_revision=True` bendros būsenos kontekste, todėl orkestratorius grąžina anksčiau veikusius agentus. Tai sukuria grįžtamojo ryšio kilpą, kurios grandinės modelis išreikšti negali.

4. **Atsiradęs bendradarbiavimas**: kadangi agentai dalijasi būsena, vėlesni agentai prisitaiko prie to, ką ankstesni įrašė. Samprotavimo agentas generuoja skirtingus promptus priklausomai nuo to, ar yra ESCO papildymo duomenų, ar nėra.


## 2. Agentų vaidmenys

### CV profiliavimo agentas
- **Įvestis**: skaito `cv_text` iš konteksto
- **Veiksmas**: naudoja LLM struktūriniam kandidato profiliui išgauti (įgūdžiai, patirtis, išsilavinimas, vyresniškumas)
- **Išvestis**: rašo `cv_profile` į kontekstą
- **Agentinis elgesys**: prisitaiko ekstrakcijos strategiją pagal CV ilgį ir formatą

### JD profiliavimo agentas
- **Įvestis**: skaito `jd_text` iš konteksto
- **Veiksmas**: dviem pakopomis — (1) klasifikuoja vaidmenį iš 135 ESCO profesijų sąrašo (su pasitikėjimo verte); (2) generuoja idealaus kandidato profilį, papildomai įtraukiant ESCO vaidmens kontekstą (scenarijuose B, C)
- **Išvestis**: rašo `jd_profile` ir `esco_role` į kontekstą
- **Agentinis elgesys**: jei ESCO vaidmens pasitikėjimas mažas, atsarginis bendrinio profesionalo režimas; struktūrinis dviejų pakopų vertinimas

### Semantinio atitikimo agentas
- **Įvestis**: skaito abu profilius
- **Veiksmas**: skaičiuoja Sentence-BERT embedding'ų kosinusinį panašumą tarp profilių santraukų + įgūdžių padengimo koeficientą
- **Išvestis**: rašo `similarity_scores` ir `skill_matches`
- **Agentinis elgesys**: vienintelis agentas sistemoje, neatliekantis LLM iškvietimo — grynai deterministinis skaičiavimas

### Samprotavimo agentas
- **Įvestis**: skaito VISĄ kontekstą (profilius, balus, ESCO papildymą, atminties įrašus jei yra)
- **Veiksmas**: LLM iškvietimu generuoja natūralios kalbos kokybės analizę — identifikuoja stiprybes, trūkumus ir abejones, siūlo pradinį balą
- **Išvestis**: rašo `reasoning_output`
- **Agentinis elgesys**: prisitaiko promptą pagal turimą kontekstą; generuoja skirtingas analizes priklausomai nuo scenarijaus

### Refleksijos agentas
- **Įvestis**: skaito samprotavimo išvestį, panašumo balus, profilius
- **Veiksmas**: vertina, ar samprotavimas nuoseklus su duomenimis; tikrina prieštaravimus ar trūkstamą analizę
- **Išvestis**: rašo `reflection_output` ir, esant reikalui, nustato `needs_revision`
- **Agentinis elgesys**: tai labiausiai „agentinis" komponentas — savarankiškai vertina darbo kokybę ir gali sukelti perskaičiavimą

### Sprendimo agentas
- **Įvestis**: skaito samprotavimą, refleksiją (jei yra), balus
- **Veiksmas**: generuoja galutinį atitikimo balą (0–100), pasitikėjimo lygį ir struktūrinį paaiškinimą; scenarijuje C — pradinį balą, kuris vėliau peržiūrimas kalibravimo agento
- **Išvestis**: rašo `final_decision` (arba `initial_decision` scenarijuje C)
- **Agentinis elgesys**: sveria kelis signalus; pritaiko pasitikėjimo vertę priklausomai nuo to, ar refleksija patvirtino ar suabejojo samprotavimu

### Atminties paieškos agentas (tik scenarijus C)
- **Įvestis**: skaito dabartinę porą + samprotavimo santrauką
- **Veiksmas**: ieško K (eksperimentuose — 5) panašiausių praeities atvejų pagal kompozitinius embedding'us
- **Išvestis**: rašo `memory_matches` su istoriniais sprendimais
- **Agentinis elgesys**: dinamiškai parenka, kurios sritys (samprotavimo, profilių, ESCO) labiausiai svarbios panašumo paieškai

### Kalibravimo agentas (tik scenarijus C)
- **Įvestis**: skaito pradinį sprendimo balą + atminties įrašus su jų etiketėmis
- **Veiksmas**: LLM iškvietimu vertina, ar pradinis balas atitinka panašių praeities atvejų tendenciją; priima vieną iš trijų sprendimų — palikti, sumažinti arba padidinti
- **Išvestis**: rašo `calibration_decision` ir galutinį pakoreguotą balą
- **Agentinis elgesys**: agentas „mokosi iš praeities" remdamasis etikečių atmintimi

### LLM-teisėjas (nepriklausomas auditas)
- **Įvestis**: CV tekstas, JD tekstas, šaltinio etiketė — **be** sistemos sprendimo
- **Veiksmas**: nepriklausomas semantinio atitikimo vertinimas; klasifikuoja į vieną iš 9 klaidų taksonomijos kategorijų
- **Išvestis**: nepriklausomas verdiktas + pasitikėjimas + klaidos kategorija
- **Agentinis elgesys**: kryžminės šeimos modelis (DeepSeek-V3.2) — nei OpenAI, nei Anthropic, nei Qwen; aklas auditas


## 3. Orkestratoriaus dizainas

Orkestratorius **nėra karkasas (framework)** — tai paprasta Python klasė, kuri:

1. Pasilieka agentų sąrašą einamajam scenarijui
2. Perduoda bendros būsenos kontekstą kiekvienam agentui paeiliui
3. Po refleksijos agento tikrina `needs_revision` vėliavą
4. Jei reikalingas perskaičiavimas IR `revision_count < max_revisions`: grąžina samprotavimo agentą
5. Žurnalizuoja visus agentų vykdymus eksperimento analizei

```
Scenarijus A: CV+JD profiliavimas → Atitikimas → Samprotavimas → Refleksija → Sprendimas
Scenarijus B: + ESCO vaidmens kontekstas JD profiliavime
Scenarijus C: + Atminties paieška + Kalibravimas (dvipakopis sprendimas)

Su refleksijos kilpa (visuose scenarijuose):
  ... → Samprotavimas → Refleksija ──→ Sprendimas
                           │
                           └─ jei needs_revision → grįžta į Samprotavimą (iki N kartų)
```


## 4. Bendros būsenos konteksto dizainas

`SharedContext` yra Pydantic modelis — vienas objektas, perduodamas kiekvienam agentui.

Dizaino principai:
- **Nekintami įvesties laukai**: `cv_text`, `jd_text`, `scenario` nustatomi vieną kartą
- **Progresyvus papildymas**: agentai prideda prie konteksto, retai perrašo
- **Pasirenkami laukai**: ESCO papildymo, refleksijos, atminties laukai yra `None` pagal nutylėjimą
- **Įmontuota žurnalizacija**: kiekvienas agento veiksmas užrašomas `logs` sąraše
- **Serializuojamas**: visas kontekstas gali būti išsaugotas į JSON vertinimui


## 5. Scenarijų palyginimas

| Funkcija                         | Scenarijus A | Scenarijus B | Scenarijus C |
|----------------------------------|:------------:|:------------:|:------------:|
| CV ir JD profiliavimas           | ✓            | ✓            | ✓            |
| Semantinis atitikimas            | ✓            | ✓            | ✓            |
| LLM samprotavimas                | ✓            | ✓            | ✓            |
| Refleksijos kilpa                | ✓            | ✓            | ✓            |
| Sprendimas                       | ✓            | ✓            | ✓            |
| ESCO vaidmens kontekstas         |              | ✓            | ✓            |
| Atminties paieška                |              |              | ✓            |
| Dvipakopis kalibravimas          |              |              | ✓            |


## 6. Kodėl tai akademiškai pagrįsta

1. **Įvardintas šablonas**: lentos paradigma yra plačiai cituojama daugiaagentės dirbtinio intelekto literatūroje. Tai nėra ad hoc dizainas.

2. **Aiški agentų autonomija**: kiekvienas agentas sprendžia, KAIP apdoroti pagal konteksto būseną, ne tik KĄ apdoroti.

3. **Refleksija = metakognicija**: refleksijos kilpa demonstruoja agentinę savęs vertinimo savybę — tai vienas iš intelektualių agentų pagrindinių bruožų.

4. **Scenarijų progresija**: A→B→C parodo augančią agentų sudėtingumą, suteikdamas natūralias eksperimentines sąlygas (ablacijos analizei).

5. **Reprodukuojamumas**: bendros būsenos žurnalizacija leidžia tiksliai pakartoti ir palyginti agentų sprendimus.

6. **Nepriklausomas auditas**: kryžminės šeimos LLM-teisėjas pateikia antrą atskaitos tašką, leidžiantį atskirti sistemos klaidas nuo duomenų rinkinio etikečių triukšmo.

7. **Ne perprojektuota**: jokio sunkaus karkaso, jokių žinučių eilių, jokių asinchroninių įvykių magistralių. Sudėtingumas atitinka bakalauro darbo apimtį.


## 7. Sąžiningi apribojimai

- Scenarijuje A vykdymas vis dar daugiausia nuoseklus. Lentos paradigma įgalina, bet nereikalauja lygiagretizmo.
- Agentų „autonomija" yra ribota — jie laikosi savo vaidmens, bet prisitaiko strategiją. Tai tinka prototipui.
- Refleksija ribota samprotavimo peržiūra, ne pilna savęs modifikacija. Tai pristatyta kaip „proof of concept" — koncepcijos įrodymas.
- LLM-teisėjo verdiktas turi savo galimas paklaidas (ilgio paklaida, sutikimo šališkumas), nors kryžminės šeimos pasirinkimas mažina pagrindines jų formas.
