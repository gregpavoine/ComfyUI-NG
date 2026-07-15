# ComfyUI-NG

## Spécification technique complète — Core moderne de génération visuelle

**Version de spécification :** 1.0
**Statut :** architecture cible initiale
**Langage principal :** Python 3.14+
**Cible initiale :** Linux x86_64, NVIDIA CUDA
**Licence recommandée :** GPL-3.0 ou AGPL-3.0 pour le core, interfaces fournisseurs séparées
**Compatibilité historique :** aucune avant FLUX.1

---

# 1. Vision du projet

ComfyUI-NG est un nouveau moteur nodal de génération d’images, inspiré du modèle d’exécution de ComfyUI, mais réécrit autour d’une architecture moderne, modulaire, fortement typée et optimisée pour :

* la latence minimale ;
* l’empreinte RAM et VRAM minimale ;
* l’utilisation maximale des ressources de l’hôte ;
* l’exécution parallèle ;
* le chargement Just-in-Time des nodes ;
* l’isolation des extensions ;
* le support exclusif des architectures modernes ;
* une API complète et stable ;
* une intégration facultative et non bloquante avec Hugging Face et Civitai.red ;
* une exploitation locale indépendante de tout service distant.

ComfyUI-NG ne doit pas être un fork lourd de ComfyUI.

Il doit reprendre les concepts utiles :

* workflow sous forme de graphe ;
* nodes composables ;
* exécution incrémentale ;
* réutilisation des résultats ;
* interface graphique nodale ;
* exécution locale ;

tout en abandonnant :

* la compatibilité Stable Diffusion historique ;
* les interfaces internes non versionnées ;
* le chargement global de tous les custom nodes ;
* le partage d’un seul environnement Python ;
* l’exécution presque entièrement centralisée dans un processus unique ;
* les dépendances implicites entre frontend, backend et extensions.

L’API actuelle de ComfyUI permet déjà de soumettre des workflows et de recevoir des événements via HTTP et WebSocket, mais elle reste étroitement liée à son format de workflow et à son serveur interne. ComfyUI-NG doit proposer une API métier indépendante, versionnée et documentée.

---

# 2. Principes non négociables

## 2.1 Local-first

Le moteur doit fonctionner intégralement :

* sans compte ;
* sans connexion Internet ;
* sans Hugging Face ;
* sans Civitai ;
* sans télémétrie distante ;
* sans API commerciale ;
* sans service cloud obligatoire.

Les intégrations distantes sont des fournisseurs optionnels.

## 2.2 Core minimal

Le processus principal ne doit contenir que :

* le serveur API ;
* le gestionnaire de graphes ;
* le scheduler ;
* le registre des nodes ;
* le gestionnaire de ressources ;
* le superviseur de workers ;
* le bus d’événements ;
* le registre local de modèles ;
* le gestionnaire de sécurité ;
* le système de configuration.

Les modèles, encodeurs, VAE, plugins et bibliothèques lourdes doivent être chargés dans des runtimes spécialisés.

## 2.3 Zéro compatibilité ancienne

ComfyUI-NG ne prend pas en charge :

* Stable Diffusion 1.x ;
* Stable Diffusion 2.x ;
* SDXL ;
* SD Turbo historique ;
* les checkpoints monolithiques SD classiques ;
* les latents 4 canaux hérités ;
* les anciens ControlNet SD1.5/SDXL ;
* les anciens workflows ComfyUI sans migration explicite ;
* les nodes V1 historiques non encapsulés.

Le modèle minimum supporté est FLUX.1 ou une architecture contemporaine équivalente.

## 2.4 Aucun réseau sur le chemin critique

Aucune génération locale ne doit attendre :

* une recherche distante ;
* une mise à jour de catalogue ;
* une vérification de version ;
* une miniature ;
* un compte utilisateur ;
* un fournisseur de modèles.

## 2.5 Isolation par défaut

Un custom node ne doit pas pouvoir :

* casser le core ;
* modifier arbitrairement l’environnement Python principal ;
* imposer une version globale de bibliothèque ;
* conserver silencieusement de la RAM ;
* ouvrir un serveur réseau sans permission ;
* accéder à des fichiers hors de ses chemins autorisés ;
* bloquer la boucle API.

## 2.6 Asynchronisme de bout en bout

Les tâches longues doivent être représentées par des jobs :

* génération ;
* téléchargement ;
* compilation ;
* indexation ;
* calcul de hash ;
* installation de plugin ;
* analyse de modèle ;
* conversion ;
* chargement en VRAM.

Aucune route HTTP ne doit rester ouverte pendant toute la durée d’une opération lourde, sauf flux événementiel dédié.

---

# 3. Modèles pris en charge

## 3.1 Familles initiales

La V1 doit viser :

* FLUX.1 Dev ;
* FLUX.1 Schnell ;
* FLUX.1 Krea ;
* FLUX.1 Fill ;
* variantes FLUX.1 compatibles ;
* FLUX.2 lorsque ses formats stables sont disponibles ;
* Qwen-Image ;
* Qwen-Image-Edit ;
* Z-Image ;
* KREA 2 ;
* modèles modernes DiT ;
* modèles modernes MMDiT ;
* modèles flow-matching ;
* modèles Transformer image futurs.

## 3.2 Registre de capacités

Le core ne doit pas tester les modèles par leur simple nom.

Chaque backend de modèle expose une description structurée :

```python
@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    family: str
    architecture: str
    task_types: frozenset[str]

    latent_channels: int
    latent_scale_factor: int
    prediction_type: str

    supported_dtypes: tuple[str, ...]
    supported_quantizations: tuple[str, ...]

    text_encoder_layout: tuple[str, ...]
    supports_negative_prompt: bool
    supports_cfg: bool
    supports_embedded_guidance: bool
    supports_img2img: bool
    supports_inpainting: bool
    supports_lora: bool
    supports_control: bool

    samplers: tuple[str, ...]
    schedulers: tuple[str, ...]
    attention_backends: tuple[str, ...]
```

## 3.3 Détection d’architecture

La détection repose sur :

* les métadonnées safetensors ;
* les noms et formes des tensors ;
* les fichiers de configuration ;
* les manifestes du dépôt ;
* les hashes connus ;
* le format de quantification ;
* les déclarations explicites du provider.

Aucune architecture ne doit être devinée à partir du nom de fichier seul.

## 3.4 Refus explicite

En cas d’ancien modèle :

```json
{
  "error": {
    "code": "unsupported_model_generation",
    "message": "Stable Diffusion 1.5 is not supported.",
    "minimum_generation": "FLUX.1",
    "detected_family": "sd15"
  }
}
```

---

# 4. Architecture générale

```text
┌──────────────────────────────────────────────────────────────┐
│                        Frontend NG                           │
│         Graph UI · Dashboard · Models · Monitoring          │
└──────────────────────────────┬───────────────────────────────┘
                               │ HTTP / WebSocket / SSE
┌──────────────────────────────▼───────────────────────────────┐
│                         API Gateway                          │
│ Auth · Validation · Rate Limits · OpenAPI · Versioning      │
└──────────────────────────────┬───────────────────────────────┘
                               │ Commands / Queries
┌──────────────────────────────▼───────────────────────────────┐
│                         NG Core                              │
│ Graph Engine · Scheduler · Registry · Event Bus · State      │
└──────────┬───────────────────┬───────────────────┬───────────┘
           │                   │                   │
┌──────────▼─────────┐ ┌───────▼────────┐ ┌────────▼──────────┐
│ Worker Supervisor  │ │ Resource Broker │ │ Provider Manager  │
│ spawn/stop/health  │ │ CPU/RAM/GPU/IO  │ │ HF/Civitai/local  │
└──────────┬─────────┘ └───────┬────────┘ └────────┬──────────┘
           │                   │                   │
┌──────────▼───────────────────▼───────────────────▼───────────┐
│                         Worker Pool                           │
│ GPU Runtime · CPU Runtime · IO Runtime · Plugin Runtime      │
└──────────────────────────────────────────────────────────────┘
```

---

# 5. Processus permanents

## 5.1 API Process

Responsabilités :

* serveur HTTP ;
* WebSocket ;
* SSE ;
* OpenAPI ;
* authentification ;
* validation des requêtes ;
* uploads ;
* téléchargements d’artefacts ;
* interrogation des jobs ;
* exposition des métriques.

Il ne charge jamais Torch, CUDA ou un modèle.

## 5.2 Core Scheduler Process

Responsabilités :

* compilation logique des graphes ;
* validation des types ;
* résolution des dépendances ;
* création du plan d’exécution ;
* calcul des priorités ;
* gestion des files ;
* coordination des workers ;
* cache des sorties de nodes ;
* annulation ;
* retry contrôlé ;
* propagation des événements.

## 5.3 Worker Supervisor Process

Responsabilités :

* création des workers ;
* arrêt des workers ;
* surveillance des heartbeats ;
* détection des processus bloqués ;
* gestion des limites mémoire ;
* isolation des plugins ;
* redémarrage contrôlé ;
* collecte des logs ;
* terminaison des arbres de processus.

## 5.4 Resource Broker

Responsabilités :

* inventaire CPU ;
* inventaire NUMA ;
* inventaire GPU ;
* budget RAM ;
* budget VRAM ;
* charge disque ;
* charge PCIe ;
* allocation des workers ;
* réservation de ressources ;
* arbitrage multi-job ;
* backpressure.

---

# 6. Architecture multiworkers

## 6.1 Types de workers

```text
GPU_MODEL_WORKER
GPU_AUX_WORKER
CPU_COMPUTE_WORKER
CPU_LIGHT_WORKER
IO_WORKER
DOWNLOAD_WORKER
METADATA_WORKER
PLUGIN_WORKER
ENCODER_WORKER
VAE_WORKER
VIDEO_WORKER futur
```

## 6.2 GPU Model Worker

Un worker GPU principal par GPU est recommandé par défaut.

Il gère :

* Transformer ;
* sampler ;
* attention ;
* LoRA ;
* compilation ;
* caches CUDA ;
* buffers de latent ;
* graphes CUDA ;
* offloading ;
* télémétrie VRAM.

Sur une seule RTX 3080 Ti, plusieurs workers CUDA exécutant simultanément de gros modèles risqueraient surtout de fragmenter la VRAM et d’ajouter des changements de contexte. La stratégie par défaut doit donc être :

```text
1 GPU lourd = 1 worker de génération principal
```

Plusieurs jobs peuvent être préparés en parallèle, mais une seule boucle de diffusion lourde utilise le GPU à la fois, sauf benchmark démontrant un bénéfice.

## 6.3 GPU Auxiliary Worker

Worker optionnel pour :

* prétraitements CUDA ;
* upscale ;
* segmentation ;
* VAE ;
* vision ;
* post-traitement.

Il peut être activé uniquement lorsque le budget VRAM le permet.

## 6.4 CPU Compute Workers

Utilisés pour :

* redimensionnement ;
* conversion d’images ;
* préparation des tensors ;
* calculs de hash ;
* analyse de fichiers ;
* compression ;
* opérations NumPy ;
* prétraitements lourds ;
* métadonnées.

Nombre par défaut :

```text
min(cœurs physiques - 2, configuration utilisateur)
```

Sur un Ryzen 9 5950X :

```text
14 workers CPU maximum conseillé
2 cœurs réservés au système, à l’API et au scheduler
```

Ce nombre doit être ajusté par benchmark, car certaines bibliothèques créent elles-mêmes des pools de threads.

## 6.5 IO Workers

Utilisés pour :

* lecture safetensors ;
* écriture des images ;
* stockage temporaire ;
* accès au cache ;
* uploads ;
* téléchargements ;
* scan des répertoires.

Ils utilisent des opérations asynchrones lorsque possible.

## 6.6 Download Workers

Indépendants du moteur de génération.

Ils gèrent :

* Hugging Face ;
* Civitai.red ;
* reprise HTTP ;
* téléchargement segmenté lorsque le fournisseur l’autorise ;
* limite de bande passante ;
* vérification de hash ;
* installation atomique ;
* annulation ;
* nettoyage des fichiers partiels.

## 6.7 Plugin Workers

Chaque plugin lourd peut disposer de son propre processus.

Politiques possibles :

```text
DEDICATED
SHARED_BY_ENVIRONMENT
SHARED_BY_TRUST_GROUP
EPHEMERAL
PERSISTENT
```

---

# 7. Multithreading

## 7.1 Règle générale

Le multithreading doit être contrôlé au niveau global.

Il faut éviter le problème classique :

```text
8 workers Python
× 16 threads OpenMP chacun
= 128 threads concurrents
```

## 7.2 Thread Budget Manager

Le Resource Broker assigne un budget par worker :

```python
@dataclass
class ThreadBudget:
    python_threads: int
    omp_threads: int
    mkl_threads: int
    torch_threads: int
    torch_interop_threads: int
```

Variables contrôlées :

```text
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
NUMEXPR_NUM_THREADS
VECLIB_MAXIMUM_THREADS
TORCH_NUM_THREADS
```

## 7.3 Pools spécialisés

* pool API asynchrone ;
* pool CPU léger ;
* pool CPU vectoriel ;
* pool IO ;
* pool hashing ;
* pool compression ;
* pool thumbnails.

## 7.4 Affinité CPU

Sous Linux, le superviseur peut attribuer :

* CPU affinity ;
* groupes de cœurs ;
* priorité nice ;
* priorité I/O ;
* réservation des cœurs du core.

Exemple pour un 5950X :

```text
CPU 0-1   : OS, API, scheduler
CPU 2-15  : CPU compute workers
CPU 16-27 : IO, metadata, encoding
CPU 28-31 : réserve et pics
```

Cette répartition reste configurable et benchmarkée.

---

# 8. Multiprocessing et Python 3.14

Python 3.14 utilise désormais `forkserver` comme méthode de multiprocessing par défaut sur plusieurs plateformes POSIX dans les usages documentés par PyTorch, ce qui est plus sûr que `fork` pour les processus complexes.

ComfyUI-NG doit explicitement imposer :

```python
multiprocessing.set_start_method("forkserver", force=True)
```

ou :

```text
spawn
```

pour les workers devant initialiser CUDA.

Le processus principal ne doit jamais initialiser CUDA avant la création des workers.

Règle :

```text
API/Core process        : pas de CUDA
GPU worker child        : initialise CUDA après spawn
CPU worker child        : aucune dépendance CUDA par défaut
```

---

# 9. Chargement JIT des nodes

## 9.1 Catalogue sans import

Au démarrage, le registre lit uniquement :

* `ng-node.toml` ;
* les schémas JSON ;
* les métadonnées ;
* la signature ;
* les permissions ;
* les dépendances ;
* les types d’entrées et sorties.

Le code Python du node n’est pas importé.

## 9.2 Manifeste

```toml
schema_version = 1

[package]
id = "org.comfyng.core-flux"
name = "ComfyUI-NG FLUX Runtime"
version = "1.0.0"
publisher = "ComfyUI-NG"
license = "GPL-3.0"

[runtime]
language = "python"
python = ">=3.14"
entrypoint = "comfyng_flux.runtime:create_runtime"
isolation = "gpu_model_worker"
unload_policy = "memory_pressure"
idle_timeout_seconds = 300

[resources]
gpu = "required"
estimated_ram_mb = 2048
estimated_vram_mb = 8000
network = false

[[nodes]]
id = "ng.model.flux.load"
display_name = "FLUX Model Loader"
input_schema = "schemas/flux_load.input.json"
output_schema = "schemas/flux_load.output.json"

[[nodes]]
id = "ng.sample.modern"
display_name = "NG Sampler"
input_schema = "schemas/sampler.input.json"
output_schema = "schemas/sampler.output.json"
```

## 9.3 Cycle de vie

```text
DISCOVERED
RESOLVED
PRELOADING
LOADED
READY
BUSY
IDLE
EVICTING
UNLOADED
FAILED
```

## 9.4 Politiques

```text
LOAD_ON_EXECUTION
PRELOAD_ON_WORKFLOW_OPEN
PRELOAD_ON_QUEUE
KEEP_WARM
UNLOAD_AFTER_EXECUTION
UNLOAD_AFTER_IDLE
UNLOAD_ON_MEMORY_PRESSURE
PERSISTENT
```

## 9.5 Déchargement réel

Un module Python chargé dans le processus principal ne peut pas être déchargé de manière fiable uniquement avec `sys.modules`.

ComfyUI-NG doit donc décharger les plugins en arrêtant leur worker.

```text
stop worker
→ fermeture des threads
→ fermeture des handles
→ libération RAM
→ destruction du contexte CUDA éventuel
→ suppression du socket IPC
```

---

# 10. Système de plugins

## 10.1 Environnements séparés

Chaque plugin dispose de :

```text
plugins/<plugin-id>/
├── manifest.toml
├── lockfile
├── package/
├── schemas/
└── .venv/
```

Gestionnaire conseillé :

```text
uv
```

## 10.2 Installation atomique

Étapes :

1. téléchargement ;
2. vérification de signature ;
3. vérification de hash ;
4. lecture des permissions ;
5. résolution des dépendances ;
6. création de l’environnement temporaire ;
7. tests d’import ;
8. validation du manifeste ;
9. déplacement atomique vers `plugins/installed` ;
10. mise à jour du registre.

## 10.3 Permissions

```toml
[permissions]
network = false
filesystem_read = ["models", "input"]
filesystem_write = ["output", "temp"]
subprocess = false
gpu = true
camera = false
microphone = false
```

## 10.4 Compatibilité V3

ComfyUI possède désormais un schéma V3 versionné pour ses custom nodes, mais cette API est encore indiquée comme évolutive. ComfyUI-NG peut fournir un adaptateur d’import, sans faire de cette compatibilité un contrat central.

## 10.5 Legacy Bridge

Un worker spécial facultatif peut exécuter certains custom nodes ComfyUI existants :

```text
comfy-legacy-bridge
```

Mais :

* il est désactivé par défaut ;
* il fonctionne dans un environnement isolé ;
* aucune garantie de performance ;
* aucune garantie de compatibilité ;
* aucune exposition directe au core ;
* aucune dépendance ancienne dans l’environnement principal.

---

# 11. Graphe d’exécution

## 11.1 Représentation

```python
@dataclass(frozen=True)
class Graph:
    id: UUID
    version: int
    nodes: tuple[NodeInstance, ...]
    edges: tuple[Edge, ...]
    inputs: Mapping[str, InputBinding]
    outputs: Mapping[str, OutputBinding]
```

## 11.2 Compilation logique

Avant toute exécution :

* validation des identifiants ;
* validation des types ;
* résolution des versions ;
* détection des cycles ;
* détection des sorties inutilisées ;
* détermination des nodes constants ;
* calcul du chemin critique ;
* estimation des ressources ;
* fusion d’opérations compatibles ;
* création de groupes d’exécution ;
* détermination des caches réutilisables ;
* affectation prévisionnelle aux workers.

## 11.3 Exécution front-to-back

Le moteur doit utiliser un tri topologique explicite.

ComfyUI a lui-même migré d’un modèle récursif vers un modèle d’exécution front-to-back fondé sur un tri topologique.

## 11.4 Sous-graphes

Support natif :

* sous-workflows ;
* fonctions de graphe ;
* macros ;
* boucles contrôlées ;
* conditions ;
* branches ;
* groupes parallèles ;
* fan-out/fan-in.

## 11.5 Types stricts

Exemples :

```text
NG_MODEL
NG_MODEL_INFO
NG_TEXT_ENCODER
NG_CONDITIONING
NG_LATENT
NG_IMAGE
NG_MASK
NG_LORA_STACK
NG_SAMPLER_CONFIG
NG_ARTIFACT
NG_JOB_REFERENCE
```

Chaque type doit disposer :

* d’un identifiant stable ;
* d’une version ;
* d’un schéma ;
* d’une stratégie de sérialisation ;
* d’une politique de transfert interprocessus.

---

# 12. Transfert interprocessus

## 12.1 Interdiction des copies inutiles

Les gros objets ne doivent pas être sérialisés en JSON ni copiés entre processus.

Mécanismes :

* mémoire partagée ;
* file descriptors ;
* Unix domain sockets ;
* shared memory ;
* fichiers temporaires mmap ;
* CUDA IPC pour cas compatibles ;
* références d’objets dans le même worker.

## 12.2 Handles

```python
@dataclass(frozen=True)
class TensorHandle:
    id: UUID
    storage: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    owner_worker: str
    byte_size: int
```

## 12.3 Localité

Le scheduler privilégie :

```text
même worker
→ même GPU
→ même domaine NUMA
→ mémoire partagée
→ transfert disque temporaire en dernier recours
```

---

# 13. Scheduler moderne

## 13.1 Objectifs

Le scheduler optimise simultanément :

* latence ;
* débit ;
* RAM ;
* VRAM ;
* énergie ;
* priorité utilisateur ;
* équité ;
* réutilisation des modèles ;
* localité des données.

## 13.2 Files

```text
interactive
normal
batch
background
download
maintenance
```

## 13.3 Priorités

```python
priority_score = (
    user_priority
    + queue_age_bonus
    + warm_model_bonus
    + cache_reuse_bonus
    - memory_pressure_penalty
    - estimated_duration_penalty
)
```

## 13.4 Coalescence

Le scheduler peut regrouper des jobs partageant :

* le même modèle ;
* le même VAE ;
* le même encodeur ;
* les mêmes LoRA ;
* la même résolution ;
* le même backend d’attention.

Le batching n’est activé que si le benchmark montre un gain réel.

## 13.5 Backpressure

Quand les ressources sont saturées :

* refus temporaire ;
* mise en queue ;
* réduction de concurrence ;
* éviction de caches ;
* suspension de jobs basse priorité ;
* limitation des téléchargements.

## 13.6 Annulation

Chaque node doit recevoir un token d’annulation.

Le sampler vérifie ce token :

* à chaque étape ;
* entre deux blocs ;
* avant le décodage ;
* avant la sauvegarde.

---

# 14. Resource Broker

## 14.1 Inventaire matériel

Le moteur détecte :

* CPU ;
* cœurs physiques ;
* threads ;
* architecture ;
* instructions SIMD ;
* NUMA ;
* RAM ;
* swap ;
* GPU ;
* VRAM ;
* compute capability ;
* version du pilote ;
* backend CUDA ;
* NVMe ;
* vitesse disque ;
* bande passante réseau ;
* température ;
* puissance ;
* charge.

## 14.2 Budgets

```yaml
resources:
  cpu:
    reserve_system_cores: 2
    max_compute_workers: auto
  memory:
    reserve_system_gb: 4
    max_pinned_gb: 8
  gpu:
    reserve_vram_mb: 768
    max_parallel_heavy_jobs: 1
  io:
    max_concurrent_reads: 4
    max_concurrent_writes: 2
```

## 14.3 Admission Control

Avant d’accepter un job en exécution :

```text
estimation modèle
+ encodeurs
+ VAE
+ LoRA
+ latents
+ buffers
+ compilation
+ marge
<= budget
```

Sinon :

* offload ;
* quantification ;
* séquençage ;
* réduction du batch ;
* refus explicite.

---

# 15. Gestion RAM et VRAM

## 15.1 Niveaux

```text
VRAM HOT
RAM PINNED
RAM NORMAL
NVME CACHE
REMOTE OPTIONAL
```

## 15.2 Cache LRU pondéré

Le coût d’éviction dépend :

* de la taille ;
* du temps de rechargement ;
* du temps de compilation ;
* de la fréquence d’utilisation ;
* du prochain job prévu ;
* du coût de téléchargement ;
* de la pression mémoire.

## 15.3 Politiques

```text
LOW_MEMORY
BALANCED
LOW_LATENCY
MAX_SPEED
MAX_THROUGHPUT
MANUAL
```

## 15.4 Pinned memory

La pinned memory est plafonnée.

Elle ne doit jamais consommer toute la RAM de l’hôte.

## 15.5 Libération ciblée

Interdit :

```python
torch.cuda.empty_cache()
```

après chaque node.

Autorisé uniquement :

* après éviction explicite ;
* après OOM récupérable ;
* après destruction d’un runtime ;
* à la demande de l’utilisateur ;
* lors d’une transition de profil.

---

# 16. Model Runtime

## 16.1 Responsabilités

* chargement de poids ;
* détection d’architecture ;
* quantification ;
* placement CPU/GPU ;
* compilation ;
* patch LoRA ;
* sampling ;
* mémoire ;
* métriques ;
* cache.

## 16.2 ModelHandle

```python
@dataclass(frozen=True, slots=True)
class ModelHandle:
    id: UUID
    family: str
    architecture: str
    local_path: Path
    sha256: str
    size_bytes: int
    source_provider: str | None
    source_model_id: str | None
    source_revision: str | None
    metadata: Mapping[str, Any]
```

## 16.3 Chargement atomique

Un modèle n’apparaît dans le registre actif que lorsque :

* tous ses fichiers existent ;
* les hashes sont valides ;
* la configuration est valide ;
* l’architecture est détectée ;
* les licences et métadonnées sont enregistrées.

---

# 17. LoRA natif

## 17.1 Nodes

```text
NG LoRA Loader
NG LoRA Stack
NG LoRA Inspector
NG LoRA Merge
```

## 17.2 Fonctions

* validation d’architecture ;
* patch du Transformer ;
* patch des encodeurs ;
* poids distincts ;
* empilement ;
* activation dynamique ;
* cache des patches ;
* fusion facultative ;
* aperçu des clés ;
* rapport des incompatibilités ;
* détection des triggers ;
* métadonnées distantes optionnelles.

## 17.3 Cache

```text
base_model_hash
+ lora_hashes
+ strengths
+ target_modules
+ dtype
+ quantization
= patched_runtime_key
```

## 17.4 Refus

Une LoRA SDXL ou SD1.5 doit être rejetée avant chargement GPU.

---

# 18. NG Sampler

## 18.1 Node principal

Entrées :

```text
model
conditioning
latent
seed
steps
guidance
denoise
sampler
scheduler
performance_profile
cache_profile
preview_profile
```

Sorties :

```text
latent
sampling_metrics
execution_trace
```

## 18.2 Samplers exposés

La liste dépend du modèle.

Aucune liste globale historique n’est affichée.

## 18.3 Scheduler exposés

Même principe :

```text
model.capabilities.schedulers
```

## 18.4 Optimisations

* préallocation ;
* buffers réutilisables ;
* cache de timesteps ;
* cache de sigmas ;
* cache de conditioning ;
* cache de bruit optionnel ;
* kernels fusionnés lorsque possible ;
* `torch.compile` ;
* CUDA Graphs conditionnels ;
* attention backend adaptatif ;
* offload asynchrone ;
* prédiction d’utilisation VRAM ;
* annulation par step ;
* télémétrie par étape.

## 18.5 Compilation

Clé :

```text
model_hash
architecture
resolution_bucket
batch
dtype
quantization
attention_backend
lora_layout
torch_version
driver_version
gpu_compute_capability
```

## 18.6 Benchmark automatique

Pour chaque couple modèle/GPU :

* SDPA ;
* Flash Attention ;
* SageAttention ;
* backend spécifique ;
* compile modes ;
* channels-last si compatible ;
* CUDA Graphs.

Le benchmark ne doit jamais être lancé automatiquement pendant une génération interactive sans consentement de configuration.

---

# 19. Nodes officiels V1

## 19.1 Modèles

```text
NG Model Loader
NG Text Encoder Loader
NG VAE Loader
NG Model Inspector
NG Model Unload
```

## 19.2 LoRA

```text
NG LoRA Loader
NG LoRA Stack
NG LoRA Inspector
```

## 19.3 Conditioning

```text
NG Prompt Encode
NG Guidance
NG Conditioning Combine
NG Conditioning Mask
```

## 19.4 Latents

```text
NG Empty Latent
NG Image To Latent
NG Latent To Image
NG Latent Resize
NG Latent Blend
```

## 19.5 Sampling

```text
NG Sampler
NG Sampler Advanced
NG Noise
NG Scheduler
```

## 19.6 Images

```text
NG Load Image
NG Save Image
NG Preview Image
NG Resize Image
NG Crop Image
NG Image Metadata
```

## 19.7 Contrôle

```text
NG Switch
NG Compare
NG Route
NG Merge
NG For Each
NG Collect
NG Subgraph Input
NG Subgraph Output
```

## 19.8 Système

```text
NG Job Info
NG Hardware Info
NG Memory Policy
NG Performance Profile
NG Cache Control
```

---

# 20. API publique

## 20.1 Base

```text
/api/v1
```

## 20.2 Domaines

```text
/system
/hardware
/health
/models
/loras
/nodes
/plugins
/workflows
/jobs
/artifacts
/cache
/providers
/downloads
/events
/config
/benchmarks
```

## 20.3 OpenAPI

Obligatoire :

```text
/api/v1/openapi.json
/docs
/redoc
```

## 20.4 Jobs

### Créer

```http
POST /api/v1/jobs
```

```json
{
  "workflow_id": "flux-portrait",
  "inputs": {
    "prompt": "portrait studio",
    "seed": 42
  },
  "execution": {
    "profile": "low_latency",
    "priority": 80,
    "keep_warm_seconds": 180
  }
}
```

### Réponse

```json
{
  "id": "01K0...",
  "status": "queued",
  "created_at": "2026-07-15T13:00:00Z"
}
```

### Lire

```http
GET /api/v1/jobs/{id}
```

### Annuler

```http
POST /api/v1/jobs/{id}/cancel
```

### Relancer

```http
POST /api/v1/jobs/{id}/retry
```

## 20.5 Événements

WebSocket :

```text
/api/v1/events/ws
```

SSE :

```text
/api/v1/events/sse
```

Types :

```text
job.created
job.queued
job.preparing
job.running
job.progress
job.preview
job.completed
job.failed
job.cancelled

node.loading
node.started
node.progress
node.completed
node.failed
node.unloaded

model.loading
model.loaded
model.offloaded
model.evicted

download.started
download.progress
download.completed
download.failed

system.memory_pressure
system.gpu_pressure
worker.started
worker.stopped
worker.crashed
```

## 20.6 Idempotence

Les routes de création acceptent :

```text
Idempotency-Key
```

## 20.7 Webhooks

```http
POST /api/v1/webhooks
```

Événements filtrables, signature HMAC et retry exponentiel.

---

# 21. Authentification

Modes :

```text
NONE_LOCALHOST
API_KEY
JWT
OIDC futur
MTLS futur
```

Configuration par défaut :

```text
localhost uniquement
aucune authentification
```

Toute écoute sur `0.0.0.0` doit produire un avertissement si aucune authentification n’est configurée.

---

# 22. Intégration Hugging Face

Hugging Face fournit un cache versionné, évite les téléchargements redondants et expose `hf_hub_download()` pour retourner directement le chemin local du fichier en cache.

## 22.1 Provider

```python
class HuggingFaceProvider(ModelProvider):
    async def search(...)
    async def inspect(...)
    async def list_files(...)
    async def resolve_download(...)
    async def download(...)
    async def authenticate(...)
```

## 22.2 Fonctions

* recherche ;
* filtres d’architecture ;
* model cards ;
* licences ;
* branches ;
* commits ;
* dépôts gated ;
* dépôts privés ;
* fichiers individuels ;
* snapshots ;
* reprise ;
* mode offline ;
* import du cache existant ;
* dry-run ;
* calcul d’espace requis.

Hugging Face permet également un mode hors-ligne via `HF_HUB_OFFLINE`, qui interdit les appels réseau et utilise uniquement le cache local.

## 22.3 Non-adhérence

Le Model Runtime ne doit jamais importer directement `huggingface_hub`.

Seul l’adaptateur provider le fait.

---

# 23. Intégration Civitai.red

## 23.1 Configuration

```yaml
providers:
  civitai_red:
    enabled: false
    adapter: civitai
    base_url: https://civitai.red
    api_base_url: ${CIVITAI_RED_API_BASE_URL}
    token: ${CIVITAI_RED_TOKEN}
```

## 23.2 Capacités

* recherche ;
* modèles ;
* versions ;
* fichiers ;
* hashes ;
* images ;
* créateurs ;
* tags ;
* type de base ;
* triggers ;
* recommandations ;
* téléchargement ;
* reprise ;
* import de métadonnées.

## 23.3 Détection de capacités

Comme les contrats précis de Civitai.red peuvent différer du service Civitai principal, le provider doit effectuer un handshake :

```json
{
  "provider": "civitai_red",
  "capabilities": {
    "search": true,
    "download": true,
    "auth": "bearer",
    "range_requests": true,
    "metadata": true
  }
}
```

Aucune URL interne ne doit être codée dans le sampler ou le registre de modèles.

---

# 24. Provider abstraction

```python
class ModelProvider(Protocol):
    id: str

    async def health(self) -> ProviderHealth: ...
    async def capabilities(self) -> ProviderCapabilities: ...
    async def search(self, query: SearchQuery) -> SearchResult: ...
    async def inspect(self, model_id: str) -> RemoteModel: ...
    async def list_files(self, model_id: str, revision: str | None) -> list[RemoteFile]: ...
    async def download(self, request: DownloadRequest, sink: DownloadSink) -> DownloadResult: ...
```

Providers initiaux :

```text
LocalFilesystemProvider
HuggingFaceProvider
CivitaiRedProvider
HTTPManifestProvider
```

---

# 25. Stockage de modèles

## 25.1 Content-addressed storage

```text
storage/
├── blobs/
│   └── sha256/<hash>
├── manifests/
├── refs/
├── metadata/
├── thumbnails/
└── partials/
```

## 25.2 Déduplication

Deux fournisseurs fournissant le même SHA-256 ne doivent créer qu’un seul blob.

## 25.3 Liens logiques

```text
models/flux/my-model.safetensors
→ blob SHA-256
```

## 25.4 Import externe

Les modèles présents dans les répertoires utilisateur peuvent être :

* référencés ;
* indexés ;
* déplacés ;
* copiés ;
* liés symboliquement ;
* liés par hardlink.

---

# 26. Base de données

SQLite en V1.

Mode :

```text
WAL
foreign_keys = ON
busy_timeout
```

Tables principales :

```text
models
model_files
model_sources
loras
plugins
plugin_versions
node_types
workflows
workflow_versions
jobs
job_events
artifacts
workers
benchmarks
provider_accounts
downloads
cache_entries
settings
```

PostgreSQL peut être ajouté plus tard pour les déploiements distribués.

---

# 27. Observabilité

## 27.1 Métriques

* durée par node ;
* attente en queue ;
* chargement modèle ;
* VRAM ;
* RAM ;
* cache hit ;
* compilation ;
* transfert CPU/GPU ;
* débit disque ;
* température ;
* puissance ;
* nombre de workers ;
* redémarrages ;
* exceptions ;
* temps d’API.

## 27.2 Prometheus

```text
/metrics
```

## 27.3 Traces

OpenTelemetry facultatif.

Trace d’un job :

```text
API
→ graph validation
→ resource admission
→ worker preload
→ text encoding
→ sampling
→ VAE decode
→ save artifact
```

## 27.4 Logs

Format JSON structuré :

```json
{
  "timestamp": "...",
  "level": "INFO",
  "component": "gpu-worker-0",
  "job_id": "...",
  "node_id": "...",
  "event": "node.completed",
  "duration_ms": 1542
}
```

---

# 28. Résilience

## 28.1 Crash plugin

Un plugin qui plante ne fait pas tomber :

* l’API ;
* le scheduler ;
* les autres workers ;
* les autres plugins.

## 28.2 Crash GPU worker

Le superviseur :

1. marque les jobs affectés ;
2. collecte les logs ;
3. détruit le worker ;
4. vérifie le GPU ;
5. redémarre le runtime ;
6. relance les jobs autorisés ;
7. évite une boucle infinie.

## 28.3 OOM

Stratégie :

1. annulation du node ;
2. libération des buffers temporaires ;
3. éviction LRU ;
4. réduction du cache ;
5. nouvel essai si autorisé ;
6. bascule offload si possible ;
7. échec clair.

---

# 29. Configuration

```yaml
server:
  host: 127.0.0.1
  port: 8188
  workers: 2

runtime:
  python: ">=3.14"
  multiprocessing_start: forkserver

scheduler:
  default_profile: balanced
  interactive_priority: 80
  max_queued_jobs: 100

cpu:
  reserve_cores: 2
  compute_workers: auto
  io_workers: 4

memory:
  reserve_system_gb: 4
  max_pinned_gb: 8

gpu:
  devices: auto
  reserve_vram_mb: 768
  heavy_workers_per_gpu: 1
  compile: auto
  attention_backend: auto

plugins:
  isolation: true
  lazy_loading: true
  default_idle_timeout: 120
  allow_legacy_bridge: false

providers:
  huggingface:
    enabled: true
    offline: false
  civitai_red:
    enabled: false
```

---

# 30. CLI

```bash
comfyng serve
comfyng doctor
comfyng benchmark
comfyng models list
comfyng models inspect
comfyng models import
comfyng models download
comfyng plugins list
comfyng plugins install
comfyng plugins disable
comfyng jobs list
comfyng jobs cancel
comfyng cache inspect
comfyng cache clean
comfyng workers status
```

---

# 31. Arborescence du dépôt

```text
comfyui-ng/
├── pyproject.toml
├── uv.lock
├── README.md
├── LICENSE
├── src/comfyng/
│   ├── api/
│   ├── core/
│   ├── graph/
│   ├── scheduler/
│   ├── resources/
│   ├── workers/
│   ├── runtime/
│   ├── models/
│   ├── lora/
│   ├── sampling/
│   ├── plugins/
│   ├── providers/
│   ├── storage/
│   ├── database/
│   ├── events/
│   ├── telemetry/
│   ├── security/
│   └── cli/
├── runtimes/
│   ├── flux/
│   ├── qwen_image/
│   ├── z_image/
│   ├── krea2/
│   └── core_image/
├── frontend/
├── schemas/
├── migrations/
├── tests/
├── benchmarks/
└── packaging/
```

---

# 32. Dépendances de base

Core :

```text
Python 3.14+
pydantic
orjson
msgspec
aiohttp ou FastAPI/Starlette
uvloop sous Linux
aiosqlite
psutil
structlog
prometheus-client
```

Runtime ML :

```text
PyTorch
safetensors
transformers minimal
tokenizers
sentencepiece si requis
Pillow
NumPy
```

Dépendances provider :

```text
huggingface_hub
httpx
```

Aucune dépendance provider dans le core minimal.

---

# 33. Frontend

## 33.1 Fonctions

* éditeur nodal ;
* recherche de nodes ;
* workflow tabs ;
* queue ;
* monitoring ;
* VRAM/RAM ;
* workers ;
* modèles ;
* LoRA ;
* téléchargements ;
* plugins ;
* logs ;
* benchmark ;
* API explorer.

## 33.2 Chargement des définitions

Le frontend récupère les nodes via :

```http
GET /api/v1/nodes
```

Il n’a pas besoin que le runtime Python du node soit chargé.

## 33.3 Validation en direct

* types ;
* compatibilité modèle ;
* LoRA ;
* VAE ;
* scheduler ;
* ressources ;
* nodes manquants.

---

# 34. Tests

## 34.1 Unitaires

* graphes ;
* scheduler ;
* capacités ;
* manifests ;
* providers ;
* cache ;
* registre ;
* API.

## 34.2 Intégration

* worker spawn ;
* crash ;
* déchargement ;
* OOM ;
* téléchargement ;
* reprise ;
* annulation ;
* génération ;
* LoRA ;
* compilation.

## 34.3 Performance

* cold start ;
* warm start ;
* RAM au repos ;
* temps d’import ;
* première image ;
* images suivantes ;
* VRAM peak ;
* temps de changement de modèle ;
* temps de déchargement ;
* débit API ;
* débit disque.

## 34.4 Soak tests

* 24 heures ;
* 1 000 jobs ;
* alternance modèles ;
* erreurs plugins ;
* interruptions réseau ;
* pression RAM ;
* pression VRAM.

---

# 35. Objectifs mesurables

Sur une installation sans plugins :

```text
API disponible                 < 1 seconde visée
Core prêt                      < 2 secondes visées
RAM core au repos              < 300 Mo visés
CUDA non initialisé au repos   obligatoire
Aucun modèle importé au boot   obligatoire
Aucun appel réseau au boot     obligatoire
```

Pour les plugins :

```text
Catalogue visible sans import       obligatoire
Déchargement RAM par arrêt worker   obligatoire
Crash isolé                         obligatoire
Environnement séparé                obligatoire
```

Pour l’API :

```text
OpenAPI complet             obligatoire
WebSocket                   obligatoire
SSE                         obligatoire
Annulation                  obligatoire
Idempotence                 obligatoire
Jobs asynchrones            obligatoire
```

---

# 36. Roadmap

## Phase 0 — Fondations

* structure du dépôt ;
* configuration ;
* logs ;
* SQLite ;
* API health ;
* worker supervisor ;
* IPC ;
* hardware probe.

## Phase 1 — Graphe et jobs

* types ;
* registre ;
* compilation de graphe ;
* scheduler ;
* files ;
* événements ;
* cache de nodes.

## Phase 2 — Runtime FLUX

* loader ;
* encodeurs ;
* VAE ;
* sampler ;
* LoRA ;
* génération T2I ;
* génération I2I.

## Phase 3 — Performance

* resource broker ;
* pinned memory ;
* offload ;
* compilation ;
* attention adaptative ;
* benchmark ;
* profiling.

## Phase 4 — Plugins

* manifests ;
* environnements ;
* workers isolés ;
* installation ;
* permissions ;
* déchargement JIT.

## Phase 5 — Providers

* local ;
* Hugging Face ;
* Civitai.red ;
* download manager ;
* CAS ;
* reprise ;
* métadonnées.

## Phase 6 — Modèles supplémentaires

* Qwen-Image ;
* Qwen-Image-Edit ;
* Z-Image ;
* KREA 2 ;
* FLUX.2.

## Phase 7 — Frontend complet

* graph editor ;
* modèles ;
* plugins ;
* monitoring ;
* benchmark ;
* API explorer.

---

# 37. Critères de validation de la V1

La V1 est considérée comme fonctionnelle lorsque :

1. le serveur démarre sans charger Torch ;
2. aucun custom node n’est importé au démarrage ;
3. un workflow FLUX peut être soumis par API ;
4. le runtime FLUX est chargé JIT ;
5. un modèle peut être maintenu chaud ;
6. un modèle peut être évincé ;
7. un plugin peut être chargé dans un worker ;
8. le worker peut être arrêté et sa RAM récupérée ;
9. une LoRA moderne peut être appliquée ;
10. le sampler choisit uniquement des options compatibles ;
11. un téléchargement Hugging Face est non bloquant ;
12. la génération locale fonctionne sans réseau ;
13. un worker qui plante ne fait pas tomber l’API ;
14. l’annulation stoppe le sampler ;
15. les métriques RAM, VRAM, CPU et temps sont accessibles ;
16. les résultats sont exposés comme artefacts versionnés ;
17. les workflows sont versionnés ;
18. l’API est documentée avec OpenAPI ;
19. l’exécution utilise plusieurs workers CPU ;
20. le GPU reste centralisé dans un runtime contrôlé.

---

# 38. Décision finale d’architecture

ComfyUI-NG doit être construit comme :

```text
un orchestrateur de graphes minimal
+ un runtime ML séparé
+ un scheduler conscient des ressources
+ des workers spécialisés
+ des plugins déclaratifs et isolés
+ une API métier stable
+ un stockage local content-addressed
+ des providers distants facultatifs
```

Il ne doit pas devenir :

```text
un gros processus Python
+ tous les plugins importés
+ toutes les dépendances dans le même venv
+ une API calquée sur le frontend
+ une liste infinie d’options historiques
```

Le cœur du projet est la séparation entre :

```text
définition d’un node
exécution d’un node
possession des données
possession du modèle
allocation des ressources
accès aux fournisseurs distants
```

Cette séparation rend possibles :

* le véritable chargement JIT ;
* le véritable déchargement ;
* l’isolation des crashs ;
* l’utilisation complète du CPU ;
* une gestion VRAM cohérente ;
* le multi-GPU futur ;
* les déploiements headless ;
* une API propre ;
* la maintenance à long terme.
