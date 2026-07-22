# Provenance — celotni zagon OpenML-CC18 na Arnes HPC

> **Status: PRED ZAGONOM.** Ta dokument je pripravljen vnaprej (predregistracija
> postopka). Rezultati (`<openml_id>.csv`, sodba, sacct potrdila) se dopolnijo
> po zagonu; mesta, ki zahtevajo vrednosti z gruče, so označena s **TODO**.
>
> Predhodnik: `results/arnes_subset/PROVENANCE.md` — validacija pilotne
> podmnožice (31/37/38), sodba ALL PASS.

## 1. Pripenjanje seznama datasetov

```bash
python scripts/gen_cc18_ids.py
```

Prebere `openml.study.get_suite(99)` (alias `OpenML-CC18`, ime "OpenML-CC18
Curated Classification benchmark"), vzame ID-je iz `suite.data`, odstrani
podvojene, uredi naraščajoče in zapiše `scripts/cc18_ids.json`. Skript se
konča z napako, če OpenML ne vrne natanko 72 datasetov.

| Postavka | Vrednost |
|---|---|
| Datoteka | `scripts/cc18_ids.json` (commitana v repo) |
| Število ID-jev | 72 (unikatnih, urejenih naraščajoče) |
| SHA-256 | `5af5ec87b7e0a774dda1f4397f4d26938f8c234ff00f4c5980e86fe92c17ece7` |
| Datum pridobitve | 2026-07-22 |
| openml verzija | 0.14.2 |

Preverjanje ob poznejšem zagonu:

```bash
sha256sum scripts/cc18_ids.json
python -c "import json; ids=json.load(open('scripts/cc18_ids.json')); print(len(ids), ids==sorted(set(ids)))"
```

Seznam se ob oddaji SLURM polja **ne** pridobiva znova — s tem je obseg
eksperimenta fiksiran in tiha sprememba zbirke na OpenML ne more neopazno
spremeniti rezultatov diplome.

### Prekrivanje s pilotom

CC18 vsebuje vse tri pilotne datasete (31 credit-g, 37 diabetes, 38 sick).
Resume logika (`run_one_dataset.py` preskoči obstoječ
`results/per_dataset/<id>.csv`) bi jih ob prisotnosti starih datotek
preskočila. Za diplomski zagon velja: **pred zagonom se
`results/per_dataset/*.csv` pobriše**, da vseh 72 datasetov izvira iz iste
verzije kode in istega okolja (enotna provenanca). Pilotni rezultati ostanejo
ločeno arhivirani v `results/arnes_subset/`.

## 2. Predpriprava (prijavno vozlišče, potrebuje internet)

```bash
python scripts/prestage.py --ids-file scripts/cc18_ids.json
```

Prenese vseh 72 datasetov v predpomnilnik OpenML in enkrat fitta TabPFN ter
TabICL, da se prenesejo uteži (računska vozlišča tečejo z `HF_HUB_OFFLINE=1`).
Neuspeli prenos posameznega dataseta ne prekine predpriprave — napake se
izpišejo na koncu, izhodna koda je 1.

Na koncu izpiše skupno velikost predpomnilnika OpenML. **Pozor:** `openml`
0.14 ignorira pripis `openml.config.cache_directory` iz `src/data.py`, zato
predpomnilnik dejansko pristane v `~/.cache/openml/org/openml/www` in ne v
`data/openml_cache`. Na gruči to pomeni, da šteje v kvoto domačega imenika
(100 GB) — preveri pred oddajo:

```bash
du -sh ~/.cache/openml ~
```

> **TODO (gruča):** vpiši dejansko velikost predpomnilnika po predpripravi
> vseh 72 datasetov in zasedenost domačega imenika.

## 3. Oddaja polja

```bash
rm -rf results/per_dataset/*           # enotna provenanca, glej 1. in 4.
sbatch scripts/run_cc18.sh
```

Briše se **vse**, ne le `*.csv`: zastarel `*.partial` iz starejše verzije kode
bi se ob nadaljevanju prebral in zastrupil rezultat.

| Direktiva | Vrednost | Utemeljitev |
|---|---|---|
| `--partition` | `gpu` | enako kot pilot |
| `--constraint` | `h100` | enako kot pilot (validirano na SM90) |
| `--account` | `fri-users` | enako kot pilot |
| `--gres` | `gpu:1` | TabPFN/TabICL |
| `--cpus-per-task` | 8 | enako kot pilot |
| `--array` | `0-71%4` | 72 ID-jev → zgornja meja 71; `%4` = throttle |
| `--time` | `08:00:00` | glej 4. |
| `--mem` | `64G` | glej 4. |

> **TODO (gruča):** potrdi throttle `%4` glede na omejitev GPU-jev na
> uporabnika in ga po potrebi zvišaj:
>
> ```bash
> sacctmgr show qos normal format=Name,MaxTRESPU%40,MaxJobsPU
> ```

> **TODO (gruča):** preveri zgornjo mejo `--time` in jo dvigni proti njej —
> manj ponovnih oddaj:
>
> ```bash
> scontrol show partition gpu | grep -i maxtime
> sacctmgr show qos normal format=Name,MaxWall
> ```

Polje je odporno na ponovni zagon na **dveh** nivojih: dataseti z že zapisanim
`results/per_dataset/<id>.csv` izpišejo `already done` in se preskočijo, dataset
prekinjen sredi dela pa se nadaljuje pri prvem nenarejenem učenju (glej 4.).

## 4. Dimenzioniranje `--time` in `--mem`

Pilotna potrdila so **spodnja meja, ne zgornja.** Pilot (job `17731379`,
dataseti 31/37/38) je tekel po ~10–60 s na task, ker so ti dataseti drobni
(največ 3772 × 29). Ta števila povedo le, da režijski stroški (nalaganje
okolja, uteži, GPU init) niso problem — ne povedo ničesar o največjih CC18
datasetih.

> **TODO (gruča):** dopolni natančna pilotna potrdila (vhod za spodnjo mejo):
>
> ```bash
> sacct -j 17731379 --format=JobID,Elapsed,MaxRSS,MaxVMSize,State,ExitCode
> ```

### Največji dataseti v CC18

Iz `python scripts/profile_datasets.py --ids-file scripts/cc18_ids.json --top 5`
(metapodatki OpenML; `atributi` = brez ciljne spremenljivke):

| ID | ime | vrstice | atributi | razredi | celice |
|---|---|---|---|---|---|
| 40927 | CIFAR_10 | 60000 | 3072 | 10 | 184.320.000 |
| 40923 | Devnagari-Script | 92000 | 1024 | 46 | 94.208.000 |
| 554 | mnist_784 | 70000 | 784 | 10 | 54.880.000 |
| 40996 | Fashion-MNIST | 70000 | 784 | 10 | 54.880.000 |
| 4134 | Bioresponse | 3751 | 1776 | 2 | 6.661.776 |

### Izbira

- **`--time=08:00:00`.** Na task se izvede 6 algoritmov × 5 foldov = 30 učenj.
  Na CIFAR_10/Devnagari sta CatBoost (privzeto 1000 iteracij) in RandomForest
  lahko po več deset minut na fold, TabPFN/TabICL pa sta na GPU dolga repa.
  8 h da velikodušno rezervo; ker je polje omejeno s throttlom in
  odporno na ponovni zagon, je predolg `--time` poceni (task se konča prej),
  prekratek pa zavrže cel task tik pred koncem.
- **`--mem=64G`.** CIFAR_10 kot `float64` zasede ~1,4 GiB na kopijo
  (60000 × 3072 × 8 B). Predobdelava (ordinalno kodiranje, imputacija),
  train/test razrez po foldih in CatBoostova kvantizacija držijo več kopij
  hkrati; 64 G da ~40× rezerve nad surovo matriko. Pilotni `MaxRSS` je pri
  tem le spodnja meja.

### Paralelizacija (popravek pred zagonom, 2026-07-22)

`RandomForestClassifier` je bil edini od štirih ansamblov brez `n_jobs` in je
zato tekel na **enem** jedru od osmih (XGBoost, LightGBM in CatBoost privzeto
uporabljajo vsa jedra, omejena z `OMP_NUM_THREADS=8`). Popravljeno z
`n_jobs=-1` v `src/models/random_forest.py`.

`n_jobs` ni hiperparameter modela, ampak nastavitev računanja — drevesa so
neodvisna, zato se rezultat ne spremeni. Preverjeno:

| Test | Izid |
|---|---|
| `predict_proba` serijsko vs. `n_jobs=-1`, sintetično 20000 × 300 in 20000 × 800 | bitno identično |
| Vseh 30 vrstic dataseta `sick` proti `results/results.csv` s popravkom | max Δ = 0.0 (vseh 6 algoritmov) |
| Pohitritev na 12 jedrih, 20000 × 300 / 20000 × 800 | 8,1× / 6,9× |
| Režija na drobnih datasetih (pilotni trije) | ~0,1–1 s na dataset |

Sodba pilotne validacije (`results/arnes_subset/PROVENANCE.md`, ALL PASS)
zaradi tega popravka **ostaja veljavna** — RF vrstice so nespremenjene.
`-1` se razreši prek joblib, ki upošteva SLURM-ovo cpuset/cgroup dodelitev,
zato pomeni 8 jeder in ne vseh jeder vozlišča.

### Kontrolne točke (dodano pred zagonom, 2026-07-22)

`src/run_one_dataset.py` po **vsakem** (algoritem, fold) učenju prepiše
`<id>.csv.partial`; končni `<id>.csv` nastane šele na koncu z atomarnim
`os.replace()`. Prekinjen task (prekoračen `--time`, preemption) torej ne
izgubi dela, nepopoln rezultat pa se ne more pretvarjati, da je popoln.

Granularnost je namenoma na nivoju posameznega učenja in ne algoritma:
najhujši realni scenarij ni "task s 6 počasnimi algoritmi", ampak **en**
algoritem, ki s svojimi 5 foldi skupaj preseže `--time` (CatBoost na
Devnagari-Script: 92000 × 1024, 46 razredov → 1000 iteracij × 46 dreves na
iteracijo). Pri kontrolnih točkah na nivoju algoritma bi tak algoritem ob
vsaki ponovni oddaji začel pri foldu 0 in se nikoli ne dokončal; pri
kontrolnih točkah na nivoju učenja pade npr. 3 folde v prvi task in 2 v
naslednjega. Partial se piše atomarno (prek `.tmp`), da ga prekinitev sredi
pisanja ne pusti okrnjenega.

**Test prekinitve in nadaljevanja (2026-07-22, lokalno, dataset 38):**

| Korak | Izid |
|---|---|
| 1. Čist enkratni zagon (referenca) | 30 vrstic |
| 2. Zagon, ubit pri 9/30 učenjih | ostane le `38.csv.partial`; končnega `38.csv` **ni** |
| 3. Ponovni zagon | "Najden partial: 9 učenj že narejenih"; dokonča na 30 vrstic |
| 4. Primerjava z referenco | 30/30 vrstic, enak vrstni red, **max Δ roc_auc = 0.0** |
| 5. Še en zagon po dokončanju | `already done` |

Pričakovano je, da TabPFN in/ali TabICL na največjih datasetih padeta na
omejitvah velikosti. To je **sprejemljivo in namerno**: fails-soft pogodba
napako zapiše v stolpec `error` posamezne vrstice, task se konča normalno,
odločitev o morebitnem poduzorčenju pa se sprejme šele, ko bo znano, kateri
dataseti dejansko odpovejo.

## 5. Po zagonu

```bash
sacct -j <jobid> --format=JobID,JobName%20,Elapsed,MaxRSS,State,NodeList   # potrdila
ls results/per_dataset/*.partial 2>/dev/null                              # nedokončani?
cp results/per_dataset/*.csv results/arnes_cc18/                          # kuriranje
python scripts/merge_results.py results/results_arnes_cc18.csv --input-dir results/arnes_cc18
python -m src.summary
```

**Preverjanje popolnosti: pričakovanih je 72 × 6 × 5 = 2160 vrstic.** Manj
vrstic ali kakršenkoli preostali `*.partial` (`merge_results.py` jih izpiše z
opozorilom in jih ne vključi v merge) pomeni dataset, ki je zadel ob steno —
dvigni `--time` in oddaj znova, nadaljeval bo, kjer je ostal. Učenje, ki
*pade*, svojo vrstico vseeno zapiše (razlog v stolpcu `error`), zato manjkajoče
vrstice vedno pomenijo "nedokončano" in nikoli "neuspešno".

Združeni CSV gre v `results/results_arnes_cc18.csv`. **Nikoli** v
`results/results.csv` — to je nespremenljiva lokalna referenca (RTX 3060).

> **TODO (gruča) po zagonu:** job ID, vozlišča, tabela Elapsed/MaxRSS po
> taskih, število vrstic (pričakovano 72 × 6 × 5 = 2160), število vrstic z
> neprazno napako in seznam datasetov, kjer je kateri algoritem odpovedal.
