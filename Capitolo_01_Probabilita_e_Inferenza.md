# Capitolo 1: Probabilità e Inferenza (*Probability and Inference*)

> **Testo di riferimento:** *Bayesian Data Analysis* (3ª Edizione), Gelman et al. (2013), pp. 3–28.

---

## 1.1 I tre passaggi dell'analisi dei dati bayesiana (*The three steps of Bayesian data analysis*)

L'inferenza statistica bayesiana consiste nel trarre conclusioni probabilistiche su quantità non osservate (parametri di una popolazione o eventi futuri) condizionatamente ai dati empirici osservati. 

Gelman et al. articolano l'intero ciclo dell'analisi dati bayesiana in **tre passaggi fondamentali**:

1. **Specificazione di un modello di probabilità completo (*Setting up a full probability model*)**:
   - Viene formulata una distribuzione di probabilità congiunta $p(\theta, y)$ per tutte le quantità osservabili ($y$) e non osservabili ($\theta$) coinvolte nel problema.
   - Il modello deve riflettere la struttura del processo di campionamento e incorporare tutta l'informazione preliminare e scientifica a disposizione sul fenomeno.
2. **Condizionamento sui dati osservati (*Conditioning on observed data*)**:
   - Si calcola la distribuzione di probabilità condizionata delle variabili non osservate subordinata ai dati raccolti: la **distribuzione a posteriori** $p(\theta \mid y)$.
   - È il cuore formale dell'applicazione della regola di Bayes: l'incertezza a priori viene aggiornata alla luce delle evidenze fornite dai dati.
3. **Valutazione dell'adeguatezza del modello (*Evaluating the fit of the model*)**:
   - Si verifica quanto bene il modello descriva i dati osservati e se le conclusioni tratte abbiano senso scientifico (*posterior predictive checks*).
   - Se emergono incongruenze tra le previsioni del modello e i dati reali, il modello viene rivalutato, ampliato o riformulato ritornando al passaggio 1.

---

## 1.2 Notazione generale per l'inferenza statistica (*General notation for statistical inference*)

- **$\theta$ (Parametri)**: Quantità di popolazione non osservate o sconosciute (vettore o matrice di parametri). Si utilizzano convenzionalmente lettere greche (es. $\theta, \mu, \sigma^2$).
- **$y$ (Dati osservati / Variabili risposta)**: Valori misurati nel campione (vettori o matrici, es. $y = (y_1, \dots, y_n)$). Si usano lettere latine minuscole per vettori e maiuscole per matrici.
- **$\tilde{y}$ (Predizioni / Dati futuri)**: Osservazioni replicabili o future non ancora rilevate, provenienti dal medesimo processo generativo.
- **$x$ o $X$ (Variabili esplicative / Covariate / Predittori)**: Informazioni note sulle unità statistiche che non vengono modellate come casuali (es. sesso, età, dosaggio). Se ci sono $k$ covariate per $n$ unità, $X$ è una matrice $n \times k$.

### Concetti cardine:
- **Scambiabilità (*Exchangeability*)**:
  Una sequenza finita di variabili aleatorie $(y_1, \dots, y_n)$ si dice *scambiabile* se la loro distribuzione congiunta $p(y_1, \dots, y_n)$ non cambia sotto qualsiasi permutazione degli indici:
  $$p(y_{\pi(1)}, \dots, y_{\pi(n)}) = p(y_1, \dots, y_n)$$
  *Significato intuitivo*: Assumere la scambiabilità equivale ad affermare che, a priori, non abbiamo motivi per distinguere un'unità osservativa da un'altra (nessuna informazione privilegiata risiede nell'etichetta dell'indice $i$).
  - I dati scambiabili sono comunemente modellati come **indipendenti e identicamente distribuiti ($iid$)** subordinatamente a un parametro incognito $\theta$:
    $$p(y \mid \theta) = \prod_{i=1}^n p(y_i \mid \theta)$$
- **Modelli gerarchici (*Hierarchical / Multilevel models*)**:
  Quando le osservazioni sono strutturate su più livelli (es. pazienti raggruppati in ospedali o studenti in scuole), la scambiabilità si applica gerarchicamente a ciascun livello (pazienti scambiabili entro lo stesso ospedale, e ospedali scambiabili tra loro).

---

## 1.3 Inferenza bayesiana (*Bayesian inference*)

### 1. La Regola di Bayes
La distribuzione congiunta dei parametri $\theta$ e dei dati $y$ si fattorizza nel prodotto tra distribuzione a priori e distribuzione campionaria:
$$p(\theta, y) = p(\theta) \, p(y \mid \theta)$$

Applicando la definizione di probabilità condizionata sui dati osservati $y$, si ottiene la **densità a posteriori**:
$$p(\theta \mid y) = \frac{p(\theta, y)}{p(y)} = \frac{p(\theta) \, p(y \mid \theta)}{p(y)}$$

dove il denominatore è la distribuzione marginale dei dati:
$$p(y) = \int p(\theta) \, p(y \mid \theta) \, d\theta \quad \text{(nel caso continuo)}$$
$$p(y) = \sum_\theta p(\theta) \, p(y \mid \theta) \quad \text{(nel caso discreto)}$$

Dato che $p(y)$ non dipende da $\theta$, agisce unicamente come costante di normalizzazione affinché l'integrale della densità a posteriori valga 1. Di conseguenza, si scrive comunemente:
$$\mathbf{p(\theta \mid y) \propto p(\theta) \, p(y \mid \theta)}$$
*La densità a posteriori è proporzionale al prodotto tra la probabilità a priori e la verosimiglianza.*

### 2. Inferenza predittiva (*Prediction*)
- **Distribuzione predittiva a priori (*Prior predictive distribution*)**:
  $$p(y) = \int p(\theta) \, p(y \mid \theta) \, d\theta$$
- **Distribuzione predittiva a posteriori (*Posterior predictive distribution*)**:
  Per una nuova osservazione futura $\tilde{y}$, assumendo l'indipendenza condizionale tra $\tilde{y}$ e $y$ dato $\theta$:
  $$p(\tilde{y} \mid y) = \int p(\tilde{y}, \theta \mid y) \, d\theta = \int p(\tilde{y} \mid \theta) \, p(\theta \mid y) \, d\theta$$
  *Intuizione*: È la media ponderata delle previsioni $p(\tilde{y} \mid \theta)$ pesate sulla distribuzione a posteriori $p(\theta \mid y)$, incorporando così l'incertezza sul parametro direttamente nella previsione!

### 3. Principio di verosimiglianza e quote a posteriori (*Odds ratio*)
- La funzione di verosimiglianza $L(\theta \mid y) = p(y \mid \theta)$ influenza l'inferenza bayesiana unicamente attraverso i suoi valori relativi per diversi $\theta$.
- Se confrontiamo due valori di parametro $\theta_1$ e $\theta_2$, il rapporto tra le densità a posteriori è espresso elegantemente tramite gli **odds**:
  $$\frac{p(\theta_1 \mid y)}{p(\theta_2 \mid y)} = \underbrace{\frac{p(\theta_1)}{p(\theta_2)}}_{\text{Prior odds}} \times \underbrace{\frac{p(y \mid \theta_1)}{p(y \mid \theta_2)}}_{\text{Likelihood ratio (Rapporto di verosimiglianza)}}$$
