# `fen_to_graph.py` — manuale

Da una riga del database dei puzzle Lichess al grafo che la rete neurale legge.

Questo documento è scritto per essere letto **prima** del codice. La prima metà è
teoria: serve a capire *perché* il codice fa quello che fa. La seconda metà è
riferimento: cosa fa ogni funzione e come si usa.

---

# Parte I — Teoria

## 1. Perché esiste questo script

Il progetto confronta due modi di far risolvere a una macchina i puzzle "matto
in n":

- una **GNN** (rete neurale su grafi), a cui diamo la posizione come grafo
- un **LLM**, a cui diamo la posizione come testo (la FEN)

La domanda di ricerca è: *per quali valori di n la GNN batte l'LLM?* E una
seconda: *aggiungere informazione temporale migliora il risultato?*

Il punto è questo. L'LLM riceve una stringa:

```
4r3/1k6/pp3P2/1b5p/3R1p2/P1R2P2/1P4PP/6K1 b - - 0 35
```

Da lì deve dedurre da solo che l'alfiere in b5 controlla f1. Deve ricostruire la
diagonale, verificare che sia sgombra, capire che quella casa è vicina al re
nemico. **Ogni volta, da zero.**

La GNN riceve un grafo in cui quel fatto **c'è già scritto**, come un arco
`b5 → f1`.

Questo script costruisce quel grafo. Non è un dettaglio implementativo: **è
l'ipotesi dell'esperimento.** La tesi del progetto è che regalare le relazioni
scacchistiche invece di farle dedurre dal testo aiuti, almeno per n piccolo. Se
l'encoder è fatto male, non c'è niente da misurare.

## 2. Perché un grafo e non un'immagine

Una scacchiera sembra fatta per una rete convoluzionale: è una griglia 8×8.
Ma c'è un problema.

In un'immagine i pixel vicini sono correlati. Sulla scacchiera **la vicinanza
fisica conta poco**: un alfiere in b5 e la casa f1 sono lontane, ma se la
diagonale è libera sono in relazione diretta e immediata.

Il grafo ribalta il punto di vista: **non modelli lo spazio, modelli le
relazioni**. b5 e f1 diventano vicini — distanza 1 — perché li colleghi con un
arco. La struttura tattica della posizione diventa la struttura del grafo.

Nel puzzle di riferimento questo non è un esempio astratto: il matto finale
funziona *proprio* perché la torre in f1 è difesa dall'alfiere in b5. Senza
quella relazione a distanza, non c'è matto.

## 3. Come funziona una rete su grafi

Serve saperlo per capire perché costruiamo il grafo così.

### Il pettegolezzo

Immagina che su ogni casa ci sia una persona. All'inizio ognuna sa una cosa
sola: sé stessa — "sono una torre bianca", "sono vuota". Ogni persona ha degli
**amici**: le case collegate a lei da un arco.

Un **giro** funziona così: *tutti contemporaneamente raccontano ai propri amici
tutto quello che sanno.*

- dopo 1 giro, ogni casa sa qualcosa dei suoi amici diretti
- dopo 2 giri, sa qualcosa anche degli amici dei suoi amici

**Un layer della rete = un giro di pettegolezzo.** Il resto — formule, pesi — è
solo il modo tecnico di dire "raccontare" e "ascoltare".

Due conseguenze pratiche:

- si aggiornano **tutte e 64 le case insieme**, ogni giro. Nessuna casa è
  speciale, nemmeno quella del re. La rete non sa dov'è il matto: deve scoprirlo.
- **troppi layer fanno male.** Se il pettegolezzo gira troppo, tutti sanno tutto
  e tutte le case finiscono con informazioni identiche. Si chiama
  *over-smoothing*. In pratica si resta fra 3 e 5 layer.

### Perché non una convoluzione

In una CNN ogni pixel ha sempre 8 vicini in posizioni fisse, quindi puoi avere un
peso per posizione. Su un grafo no: un nodo può avere 2 vicini o 27, e non
esistono "il primo vicino". Serve un'aggregazione che non dipenda dall'ordine e
regga un numero variabile di input — somma, media o massimo. È tutto il trucco.

## 4. Nodi, archi, e come si scrivono

**Nodo** = una casa. Sono 64, numerate `a1=0, b1=1, ..., h8=63`.

Le case sono scritte in fila, otto per traversa:

```
traversa 8 | 56 57 58 59 60 61 62 63
traversa 7 | 48 49 50 51 52 53 54 55
    ...
traversa 1 |  0  1  2  3  4  5  6  7
             a  b  c  d  e  f  g  h
```

Da cui le due operazioni che vedrai nel codice:

- `square // 8` → la **traversa** (in inglese *rank*)
- `square % 8` → la **colonna** (in inglese *file* — falso amico, niente a che
  vedere con i file del computer)

**Arco** = una relazione fra due case. Si scrivono in una matrice `2 × E` da
leggere **in verticale**: ogni colonna è una freccia.

```
riga 0 (da):    6    6    6    6 ...
riga 1 (a):     5    7   13   14 ...
                ↑
        colonna 0 = una freccia dal nodo 6 al nodo 5, cioè g1 → f1
```

Sembra scomodo ma è il formato che vuole PyTorch Geometric, perché memorizza
**solo le frecce che esistono**. L'alternativa — una griglia 64×64 — avrebbe
4096 celle di cui, in una posizione tipica, solo un'ottantina diverse da zero.
Su milioni di puzzle sono gigabyte di zeri.

**La direzione conta.** La colonna `(6, 5)` significa "da g1 a f1", e non è la
stessa cosa di `(5, 6)`. L'informazione scorre solo nel verso della freccia.

## 5. Le semi-mosse

Nel linguaggio comune degli scacchi **una mossa comprende le azioni di entrambi
i giocatori**: `1. e4 e5` è *la mossa 1*.

Per il codice serve un'unità più fine, la **semi-mossa** (in inglese *ply*):

> Una semi-mossa è una singola azione di un singolo giocatore.

Una mossa = due semi-mosse. Nient'altro.

Nel dataset il campo `Moves` di un `mateIn2` è:

```
e5f6  e8e1  g1f2  e1f1
```

Sono **quattro semi-mosse**. Da qui l'invariante che il codice usa dappertutto:

```
len(Moves) = 2n
```

## 6. Il tempo — due meccanismi distinti

Il testo del progetto chiede di incorporare informazione temporale. I puzzle
Lichess **non hanno tempi**, quindi vanno simulati. Lo script ne produce due,
diversi e in posti diversi.

| dove | nome | cosa rappresenta |
|---|---|---|
| nodi | `think_time` | quanto è difficile questa posizione |
| archi | `edge_time` | da quante semi-mosse si è mosso quel pezzo |

### `think_time` — difficoltà

Una prima versione era `Rating / 3000`. Non funzionava, per un motivo che vale la
pena capire: **una funzione monotona del rating è il rating**. La correlazione è
1.00, quindi l'ablation "con tempo vs senza tempo" misurava in realtà "con
rating vs senza rating".

E c'era un secondo problema: quel valore era identico su tutti gli esempi dello
stesso puzzle. Un matto in 1 e un matto in 5 con lo stesso rating ricevevano lo
stesso "tempo di riflessione". Niente di temporale.

La formula attuale combina tre fattori:

- il **rating** — puzzle difficile, si pensa di più
- `n_remaining` — sulla prima mossa cerchi l'idea, sulle ultime esegui
- il **numero di mosse legali** — più opzioni, più tempo a scartarle

Gli ultimi due variano *dentro* lo stesso puzzle. La correlazione col rating
scende a **0.74**, e su un `mateIn5` il valore decade `0.385 → 0.160` man mano
che il matto si avvicina.

I coefficienti sono **tarati a mano**, non misurati. È un proxy di difficoltà, e
va dichiarato come tale nella relazione.

### `edge_time` — recency

Il modello del prof applica un decadimento esponenziale all'attenzione:

```python
decay = exp(-lambda_decay * delta_t)
alpha = alpha * decay
```

`delta_t` è **per arco**. Nel loro dominio è ovvio: un arco è il passaggio da un
evento al successivo, quindi ha una durata. Nel nostro un arco è una relazione
dentro un'unica fotografia: **non c'è nessuna durata**.

La recency è il ponte. Ogni arco è generato dal pezzo sulla casa di partenza,
quindi eredita **da quante semi-mosse quel pezzo si è mosso**.

```
l'alfiere in b5 si è mosso 2 semi-mosse fa
   ↓
l'arco  b5 → f1  porta delta_t = 2
```

Perché è utile: quando guardi una posizione la prima cosa che cerchi è *cosa è
appena cambiato*. Il pezzo appena mosso è quasi sempre quello che ha creato la
minaccia. E nei puzzle Lichess la mossa dell'avversario è **per costruzione
l'errore che apre la tattica**.

> **Attenzione, questo è il punto in cui si sbaglia più facilmente.**
> `lambda_decay` è un float fisso che passate voi, e il layer non normalizza
> niente. Nel notebook degli autori vale `0.01`, tarato su durate in ore. Con
> recency fra 0 e 20 quel valore fa variare il decadimento del 10%: il
> meccanismo è di fatto spento. **Serve un lambda fra 0.1 e 0.2.**
> Copiare `0.01` e poi concludere "il tempo non aiuta" significa aver misurato
> la propria taratura, non il tempo.

---

# Parte II — Le trappole del dataset

Tre cose che il codice gestisce e che, se ignorate, rovinano i dati **senza dare
nessun errore**.

## 1. La FEN non è la posizione del puzzle

È la posizione **un istante prima**. La prima mossa di `Moves` è la mossa
dell'**avversario**, quella che innesca il puzzle.

```
FEN nel csv        : ... 6K1 w - - 0 35      <- tocca al BIANCO
posizione puzzle   : ... 6K1 b - - 0 35      <- tocca al NERO
```

Nel puzzle di riferimento il matto lo dà il **nero**, non il bianco. Prendendo la
FEN così com'è si insegna al modello a cercare il matto per il colore sbagliato,
dalla posizione sbagliata. Nessun crash, solo un modello che non impara.

`get_puzzle_position` applica quella mossa. **Non va applicata anche a monte**:
se la applicano in due, la posizione salta di una mossa.

## 2. Le promozioni

La label è `y = from*64 + to`, che non distingue `e7e8q` (donna) da `e7e8n`
(cavallo): entrambe danno `3388`.

La convenzione adottata è **promozione = donna**. Le righe dove il *solver*
sotto-promuove porterebbero una label sbagliata in silenzio, quindi vengono
scartate. Misurate sul database: **1.431 su 6,1 milioni, lo 0,023%**, quasi tutte
promozioni a cavallo con scacco. Una sottopromozione dell'*avversario* non
diventa mai una label ed è innocua.

Le promozioni a donna restano: sono **147.699** e la convenzione le rende non
ambigue.

> In fase di valutazione il decoder deve **riaggiungere la donna**. Una mossa
> `b7b8` senza il pezzo non esiste: `python-chess` la rifiuta come illegale, e
> conteresti come errori 147.699 risposte corrette.

## 3. Le spinte dei pedoni

`board.attacks()` restituisce le case che un pezzo **controlla**. Per i pedoni
sono solo le due diagonali — dove possono *mangiare* — non la casa davanti.

```
pedone e2
  attacks(e2) : ['d3', 'f3']       <- dove potrebbe mangiare
  mosse legali: ['e2e3', 'e2e4']   <- dove può davvero andare
```

Senza correzione il grafo avrebbe archi verso case dove il pedone non può
muovere, e nessun arco verso quelle dove può. Da qui il quarto tipo di arco,
`pushes`, generato a mano da `pawn_pushes`.

Nota: per un pedone un arco `moves` verso una diagonale vuota significa
**controllo**, non una mossa legale. È tenuto così di proposito: quella
relazione conta per il matto (il re nemico non può andarci).

---

# Parte III — I dati

## Cosa entra

Un dizionario o una riga di pandas. Lo script **non legge il CSV**: quello lo fa
lo script di filtro e split, a monte.

| campo | tipo | a cosa serve |
|---|---|---|
| `FEN` | stringa | la posizione, un istante prima del puzzle |
| `Moves` | lista o stringa | mosse UCI; la prima è dell'avversario |
| `Rating` | numero o stringa | difficoltà, usata per `think_time` |
| `Themes` | lista o stringa | temi Lichess |
| `MateIn` | numero | profondità, usata solo come controllo incrociato |
| `PuzzleId` | stringa | per verificare il leakage fra split |

## Cosa esce

Una **lista** di esempi — uno per ogni decisione di chi risolve. Lista vuota se
la riga è inutilizzabile.

### Quello che la rete legge

| campo | forma | contenuto |
|---|---|---|
| `x` | (64, 12) float32 | un nodo per casa. Con `use_timing=True` diventa (64, 13) |
| `edge_index` | (2, E) int64 | riga 0 = partenza, riga 1 = arrivo |
| `edge_attr` | (E, 4) float32 | one-hot: `attacks`, `defends`, `moves`, `pushes` |
| `edge_time` | (E,) float32 | la recency per arco. `None` senza timing |

Le 12 feature dei nodi, in ordine:

```
pawn, knight, bishop, rook, queen, king,   ← che pezzo (one-hot)
white, black,                              ← di che colore
empty,                                     ← casa vuota
file, rank,                                ← dove sta, normalizzato in [0,1]
white_to_move                              ← a chi tocca (uguale su tutti i nodi)
```

### Quello che la rete deve indovinare

| campo | tipo | contenuto |
|---|---|---|
| `y` | int 0-4095 | la mossa, come `from_square * 64 + to_square` |
| `legal_moves` | array int16 | le mosse legali, stessa codifica |

`legal_moves` serve a **mascherare l'uscita**. La rete produce 4096 punteggi ma
in una posizione tipica le mosse legali sono una trentina: azzerando le altre
prima dell'argmax, l'accuratezza misura la bravura scacchistica e non quanto il
modello ha imparato le regole.

È anche la baseline da mettere nella relazione: tirare a caso fra le mosse legali
dà circa **1 su 26 = 3,8%**, contro 1 su 4096 = 0,02% senza maschera. Se il
modello fa il 4%, non ha imparato niente.

Le mosse legali sono salvate come **lista di indici** e non come maschera da 4096
booleani: la maschera costerebbe 4 KB per esempio, cioè gigabyte su milioni di
esempi. La lista sono trenta numeri, e la maschera si ricostruisce in due righe.

### Contabilità, per te e non per la rete

| campo | contenuto |
|---|---|
| `n_remaining` | profondità vera **di questa posizione** |
| `puzzle_n` | profondità del puzzle intero |
| `puzzle_id` | per verificare il leakage fra split |
| `rating`, `think_time` | per le analisi |
| `fen` | per ricostruire la posizione |
| `solution` | la linea completa, serve in valutazione |

> `n_remaining` **non** coincide con `puzzle_n`. Srotolando un `mateIn5` ottieni
> cinque esempi di profondità 5, 4, 3, 2, 1. Il grafico "accuratezza per n" va
> costruito su `n_remaining`, non sul tema della riga.

---

# Parte IV — Le funzioni

## `normalize_row(row) -> dict`

Uniforma i tipi in ingresso. Da qui in poi il resto del codice sa cosa ha in mano.

Lavora su una **copia**, quindi la riga di chi chiama resta intatta. È idempotente:
chiamarla due volte non rompe niente.

Il bug che previene: se il collega usa `pandas.read_csv`, `Moves` arriva come
stringa. E `"e5f6 e8e1"[0]` è `'e'`, la lettera — non la mossa. L'errore che ne
segue è pure fuorviante, perché `push_uci('e')` fallisce e il `try/except` lo
etichetta come "Invalid FEN" su una FEN perfettamente valida.

## `get_puzzle_position(row) -> (board, solution)`

Costruisce la scacchiera e ci gioca la mossa dell'avversario. Restituisce la
posizione vera e le mosse rimanenti. Su FEN o mossa non valida restituisce
`(None, None)`.

`push_uci` verifica anche la legalità: è un controllo di qualità gratis.

Nota: `push` modifica la scacchiera **sul posto**. Per rigiocare una linea senza
distruggere la posizione serve `board.copy()`.

## `pawn_pushes(board, square, piece) -> list`

Le case dove un pedone può avanzare: una in avanti, più il doppio passo dalla
traversa di partenza, entrambe solo se libere.

Il secondo controllo è **annidato** dentro il primo, e non è un caso: un pedone
non può scavalcare, quindi se la casa davanti è occupata il doppio passo è
illegale anche se la seconda fosse vuota.

Calcolate a mano invece che da `board.legal_moves`, perché quello copre solo il
colore che deve muovere e noi vogliamo entrambi — coerentemente con `attacks()`.

## `build_node_features(board) -> (64, 12)`

Compila la scheda anagrafica di ogni casa. La riga di e8 nel puzzle di
riferimento si legge: *"c'è una torre, è nera, colonna e, ottava traversa, tocca
al nero."*

Il `/ 7.0` su colonna e traversa non è estetica: senza, quelle feature andrebbero
da 0 a 7 mentre le altre stanno fra 0 e 1, dominerebbero i gradienti e
l'addestramento diventerebbe instabile.

`white_to_move` è identico su tutti e 64 i nodi. È voluto: la rete guarda solo i
nodi e non ha un posto dove leggere un'informazione globale.

## `build_edge(board) -> (edge_index, edge_attr)`

La rubrica telefonica: chi parla con chi. Per ogni pezzo guarda le case che
controlla e classifica in base a cosa c'è dall'altra parte:

- casa libera → `moves`
- pezzo avversario → `attacks`
- pezzo proprio → `defends`

Più le spinte dei pedoni → `pushes`.

Il blocco dei pedoni sta **dentro** il ciclo sulle case ma **fuori** da quello
sugli attacchi. Se finisce dentro, ogni spinta viene aggiunta una volta per
casa attaccata: duplicati. C'è un test apposta.

Sul puzzle di riferimento: **86 archi**, 1 `attacks`, 12 `defends`, 63 `moves`,
10 `pushes`.

## `build_label(solution) -> int`

La mossa da indovinare, compressa in un numero: `from_square * 64 + to_square`.
64 partenze × 64 arrivi = 4096 possibilità, e `y` dice quale è quella giusta.

Si torna indietro con `y // 64` e `y % 64`, e servirà davvero in valutazione.

`chess.Move.from_uci` legge anche il pezzo promosso, ma il codice usa solo
`from_square` e `to_square`: il pezzo viene scartato di proposito, secondo la
convenzione "promozione = donna".

## `build_legal_moves(board) -> array`

Le mosse legali nella stessa codifica di `y`. `np.unique` perché le quattro
promozioni della stessa casa collassano sullo stesso indice.

## `simulate_think_time(rating, n_remaining, n_legal) -> float`

Il tempo di riflessione simulato, normalizzato in `[0, 1]`.

```python
base   = 2.0 + 18.0 * (rating - 400) / 2600.0   # ~2..20 secondi
depth  = 1.0 + 0.35 * (n_remaining - 1)         # moltiplicatore
branch = 1.0 + 0.02 * (n_legal - 20)            # moltiplicatore
```

`base` è l'unica quantità con un'unità di misura; le altre due sono numeri puri
attorno a 1 che la stirano. Si moltiplica invece di sommare così gli effetti si
compongono e restano proporzionali.

Il taglio a un minuto evita che qualche caso estremo schiacci tutti gli altri
verso lo zero.

## `build_edge_time(edge_index, last_moved, played) -> array`

Per ogni arco, la recency del pezzo sulla **casa di partenza**. Chi non è nel
registro prende `UNKNOWN_RECENCY = 20`.

## `update_last_moved(last_moved, move, played)`

Aggiorna il registro dopo una mossa. Due regole:

1. **cancella** la voce della casa di partenza — quel pezzo non è più lì
2. **scrive** la casa d'arrivo con la semi-mossa corrente

La regola 2 gestisce le catture da sola: la voce del pezzo mangiato viene
sovrascritta.

## `row_to_graph(row, use_timing=False, unroll=True) -> list`

La funzione pubblica. Non calcola quasi niente: chiama le altre nell'ordine
giusto e impacchetta.

**I quattro controlli**, dal più economico al più caro. Se uno fallisce, lista
vuota:

1. `MateIn` coerente con `len(Moves) // 2`
2. nessuna sottopromozione fra le mosse del solver
3. FEN e mossa dell'avversario valide
4. l'intera linea si rigioca fino in fondo su una copia

Il quarto è importante: `get_puzzle_position` protegge solo `Moves[0]`, e una
mossa illegale più avanti farebbe saltare l'intero preprocessing su una riga
corrotta. Rigiocare costa qualche `push` e succede **prima** di costruire i
grafi.

**Il ciclo.** Scorre la soluzione e produce un esempio ogni volta che tocca a chi
risolve. Le mosse dell'avversario si giocano e basta: non sono decisioni.

```
e8e1  →  tocca al nero, è il solver  →  ESEMPIO 1
g1f2  →  tocca al bianco             →  la gioca e basta
e1f1  →  tocca al nero               →  ESEMPIO 2
```

Nessun riferimento a `n`: funziona per matto in 1 come per matto in 10.

---

# Parte V — Come si usa

## Installazione

```bash
pip install chess numpy
```

Nient'altro. Nessun torch, nessun torch-geometric: l'encoder è puro numpy.

## Uso base

```python
from fen_to_graph import row_to_graph

row = {
    "PuzzleId": "000Zo",
    "FEN": "4r3/1k6/pp3r2/1b2P2p/3R1p2/P1R2P2/1P4PP/6K1 w - - 0 35",
    "Moves": "e5f6 e8e1 g1f2 e1f1",
    "Rating": 1363,
    "Themes": "endgame mate mateIn2 operaMate short",
    "MateIn": 2,
}

examples = row_to_graph(row, use_timing=True)

for e in examples:
    print(e["y"], e["n_remaining"], e["x"].shape, e["edge_index"].shape)
```

```
3844  2  (64, 13)  (2, 86)
261   1  (64, 13)  (2, 89)
```

## I due flag

| flag | `True` | `False` |
|---|---|---|
| `unroll` | un esempio per ogni mossa del solver | solo la prima mossa |
| `use_timing` | aggiunge `think_time` ai nodi e `edge_time` agli archi | senza |

Le quattro combinazioni servono tutte:

```python
train = row_to_graph(row, unroll=True,  use_timing=True)    # con tempo
train = row_to_graph(row, unroll=True,  use_timing=False)   # senza tempo (ablation)
test  = row_to_graph(row, unroll=False, use_timing=True)    # valutazione
```

## Su un CSV intero

```python
import pandas as pd

df = pd.read_csv("train.csv")
scarti = 0
dataset = []

for _, row in df.iterrows():
    esempi = row_to_graph(row, use_timing=True, unroll=True)
    if not esempi:
        scarti += 1
        continue
    dataset.extend(esempi)

print(f"{len(dataset)} esempi da {len(df)} righe, {scarti} scartate")
```

Conviene tenere contatori **separati per motivo di scarto**. Se ne scarti una su
un milione è rumore; se ne scarti duecentomila hai un bug sistematico a monte, e
devi accorgertene subito e non tre settimane dopo.

## Verso PyTorch Geometric

L'encoder produce array numpy. Il modello vuole tensori torch:

```python
import torch
from torch_geometric.data import Data

def to_pyg(e):
    return Data(
        x=torch.from_numpy(e["x"]),
        edge_index=torch.from_numpy(e["edge_index"]),
        edge_attr=torch.from_numpy(e["edge_attr"]),
        y=torch.tensor(e["y"]),
    )
```

`edge_time` va passato separatamente come argomento `time` del layer, non dentro
`Data`.

## L'ordine da rispettare

> **Prima lo split train/val/test, poi lo srotolamento.**

Gli esempi che escono dallo stesso puzzle sono posizioni quasi identiche — la
stessa partita a due semi-mosse di distanza. Se finiscono in split diversi, il
modello in test riconosce qualcosa che ha già visto, l'accuratezza sale e il
risultato non vale niente. Si chiama *data leakage*.

Verifica, tre righe:

```python
assert not (set(train.PuzzleId) & set(test.PuzzleId))
```

Ed è il motivo per cui ogni esempio porta con sé `puzzle_id`.

---

# Parte VI — I test

Il file contiene un main con **75 controlli**. Si lancia direttamente:

```bash
python fen_to_graph.py
```

Ogni controllo stampa `PASS` o `FAIL`, e non si ferma al primo errore: vedi tutti
i problemi in un colpo solo.

I valori attesi non sono inventati: vengono dal puzzle `000Zo`, verificati sulla
scacchiera. Quelli che vale la pena conoscere:

| controllo | cosa protegge |
|---|---|
| `BLACK is to move, not white as in the FEN` | la trappola di Lichess. Se qualcuno toglie il `push_uci`, diventa rosso subito |
| `the solution really is mate` | rigioca le mosse e usa python-chess come giudice indipendente |
| `every edge type matches the board` | ricontrolla tutti e 86 gli archi, non tre esempi |
| `no duplicated push edges` | l'indentazione del blocco dei pedoni |
| `think_time differs between the two examples` | che la feature del tempo sia davvero temporale e non un rating riscalato |
| `example 1: the only piece with a history is f6` | l'offset delle semi-mosse nel registro |
| `the departed square e8 no longer carries a recency` | la regola 1 di `update_last_moved` |

---

# Parte VII — Limiti noti

Tutti da dichiarare nella relazione. Un limite scritto è metodologia; un limite
taciuto è un errore.

**Archi monodirezionali.** L'informazione scorre solo nel verso della freccia:
un pezzo sa chi lo attacca, non cosa attacca. Per il riconoscimento del matto la
direzione scelta è quella giusta — la casa del re deve sapere chi la minaccia —
ma un pezzo resta cieco sui propri bersagli, il che pesa sui sacrifici e sulle
deviazioni. Aggiungere `attacked_by` / `defended_by` è un esperimento pulito, e
l'ipotesi ragionevole è che aiuti soprattutto per n grande.

**Recency rada al primo esempio.** Si conosce solo `Moves[0]`, quindi nel puzzle
di riferimento sono 3 archi su 86. Attenuante seria: quella mossa è l'errore che
innesca il puzzle, cioè il fatto più informativo che ci sia. Con i PGN
raggiungibili dalla colonna `GameUrl` il registro partirebbe pieno.

**Arrocco ed en passant.** L'arrocco muove due pezzi ma `python-chess` riporta una
mossa sola, quindi la torre non viene registrata. L'en passant cattura un pedone
che non sta sulla casa d'arrivo, e la sua voce resta nel registro. Entrambi rari
nei matti.

**Il tempo è simulato, non misurato.** I coefficienti sono tarati a mano perché
somiglino a tempi umani plausibili. L'ablation dimostra se un proxy di difficoltà
aiuta, non se il tempo umano aiuti. Vale la pena fare un'analisi di sensibilità:
se la conclusione regge a due o tre parametrizzazioni diverse, non dipende dai
numeri scelti.

**Equità del confronto con l'LLM.** La FEN non contiene l'ultima mossa giocata.
Se la GNN conosce la recency e l'LLM no, il confronto pende. Va usato il modello
senza timing per quel confronto, oppure dichiarata l'ultima mossa anche all'LLM.
Scegliete e scrivetelo.

---

# Parte VIII — Cosa manca

L'encoder è completo. Quello che resta vive in altri file:

1. **Adattatore verso PyG** — da numpy a tensori torch, con `edge_time` passato
   come argomento `time`
2. **Driver sul CSV** — cicla, conta gli scarti per motivo, salva
3. **Decoder per la valutazione** — da `y` a `chess.Move`, aggiungendo
   `promotion=QUEEN` quando un pedone raggiunge l'ultima traversa
4. **Taratura di `lambda_decay`** — 0.1-0.2, non lo 0.01 del notebook
5. **Verifica del leakage** e distribuzione degli esempi per `n_remaining`
