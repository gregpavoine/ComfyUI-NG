# ComfyUI-NG — Design d’architecture V1

**Statut :** approuvé par la demande utilisateur et la spécification technique 1.0  
**Autorité normative :** `docs/SPECIFICATION.md`  
**Cible :** Linux x86_64 avec NVIDIA CUDA ; développement et tests du control-plane possibles sur macOS sans CUDA  
**Version minimale :** Python 3.14

## 1. Résultat livré

ComfyUI-NG est livré comme un monorepo local-first exploitable de bout en bout : API métier versionnée, moteur de graphes strictement typé, scheduler conscient des ressources, workers isolés, stockage content-addressed, registre de modèles, plugins déclaratifs, providers optionnels et interface nodale React. Le chemin de génération réel charge Torch et les pipelines ML uniquement dans un worker de runtime, jamais dans le serveur API ni dans le scheduler.

La V1 ne prétend pas émuler les anciens workflows ComfyUI. Elle refuse explicitement SD1.x, SD2.x et SDXL, et ne publie que les options déclarées par les capacités du modèle actif.

## 2. Choix d’architecture

### Option retenue — monorepo Python/React à processus isolés

- Python 3.14 pour le control-plane, FastAPI pour HTTP/WebSocket/SSE/OpenAPI, SQLite WAL pour l’état local.
- Processus distincts pour API, scheduler/superviseur et runtimes ; `forkserver` sous Linux et `spawn` sur les plateformes qui l’exigent.
- IPC par messages typés et handles ; blobs volumineux placés dans le CAS ou la mémoire partagée, jamais encodés en JSON dans les commandes internes.
- React + TypeScript + React Flow pour l’éditeur nodal ; les définitions proviennent uniquement de `GET /api/v1/nodes`.
- Runtime ML facultatif installé par extra, avec import tardif de PyTorch/diffusers dans le worker GPU.

Ce choix est le plus fidèle à la spécification, garde une installation locale simple et permet d’exécuter et tester le control-plane sans GPU.

### Option écartée — control-plane Rust, runtimes Python

Elle réduirait encore la RAM et renforcerait l’isolation, mais introduirait une seconde chaîne de compilation, complexifierait les contrats IPC et s’écarterait du langage principal demandé.

### Option écartée — microservices conteneurisés systématiques

Elle préparerait mieux un déploiement distribué, mais augmenterait fortement le temps de démarrage et le coût opérationnel d’un produit local-first. Les frontières de processus retenues restent transformables en services ultérieurement.

## 3. Limites de processus

```text
Browser
  -> API process (FastAPI, auth, OpenAPI, static frontend)
      -> command/event channel
          -> scheduler + resource broker
              -> worker supervisor
                  -> CPU / IO / plugin / GPU runtime processes
```

Le processus API ne dépend pas de Torch, CUDA, diffusers, transformers ni huggingface_hub. Les adapters providers résident dans des modules optionnels importés seulement lors de leur activation. Les runtimes possèdent leurs modèles et tensors ; le core ne manipule que `ModelHandle`, `TensorHandle`, `ArtifactReference` et des messages sérialisables.

## 4. Modules et responsabilités

### Control-plane

- `config` : chargement YAML + variables d’environnement, validation stricte et valeurs par défaut sûres.
- `database` : migrations idempotentes, connexions SQLite WAL, repositories transactionnels.
- `events` : bus asynchrone, journal durable des événements, fan-out WebSocket/SSE/webhooks.
- `graph` : types, validation, détection de cycles, tri topologique, compilation et estimation.
- `scheduler` : six files, score de priorité, admission, backpressure, annulation et retry borné.
- `resources` : inventaire CPU/RAM/disque/GPU, budgets, réservations, pression mémoire et thread budgets.
- `workers` : protocole IPC, heartbeats, spawn/stop, limites, crash recovery et terminaison des arbres.
- `security` : auth localhost/API key/JWT, permission sets, validation de chemins et signatures HMAC.

### Données et extensions

- `storage` : CAS SHA-256, imports atomiques, refs logiques, artefacts versionnés, fichiers partiels et déduplication.
- `models` : registre, inspection safetensors/config, détection par structure et refus des familles historiques.
- `lora` : métadonnées, compatibilité de famille, stack/clé de cache et commandes de patch runtime.
- `plugins` : découverte manifeste sans import, validation, installation atomique, environnements séparés et workers isolés.
- `providers` : protocole commun et adapters local, HTTP manifest, Hugging Face et Civitai.red avec handshake.

### Runtime ML

- Une factory de runtime reçoit un modèle validé et choisit le backend d’après ses capacités, pas son nom de fichier.
- Les familles FLUX.1/FLUX.2, Qwen-Image, Qwen-Image-Edit, Z-Image, KREA 2 et les DiT/MMDiT compatibles passent par un adapter diffusers moderne lorsqu’un pipeline local compatible est installé.
- Le runtime implémente T2I, I2I, inpainting lorsque la capacité l’autorise, LoRA, sampling, previews, annulation par étape, keep-warm, éviction et métriques.
- Un runtime de test déterministe existe uniquement sous `tests/` afin de valider le chemin complet sans télécharger de poids.

### Frontend

- Shell sombre, dense et lisible, inspiré d’un poste de contrôle GPU plutôt que d’un tableau de bord générique.
- Éditeur React Flow avec palette recherchable, connexions typées, validation immédiate, minimap, zoom, sélection, duplication et suppression.
- Onglets de workflows et persistance serveur ; inspector contextuel des entrées/sorties.
- Surfaces fonctionnelles pour queue, jobs, modèles, LoRA, plugins, providers, téléchargements, workers, mémoire, logs, benchmarks et API explorer.
- Mise à jour temps réel par WebSocket avec repli SSE ; état dégradé explicite si le backend est indisponible.

## 5. Flux de données principal

1. Le frontend charge le catalogue déclaratif et un workflow versionné.
2. `POST /api/v1/jobs` valide l’idempotence, les entrées et l’existence de la version du workflow, puis répond immédiatement `queued`.
3. Le scheduler compile le graphe, vérifie types/cycles/options de modèle, estime les ressources et demande une réservation.
4. Le superviseur démarre ou réutilise les workers nécessaires ; le runtime GPU charge le modèle JIT.
5. Les nodes s’exécutent dans l’ordre topologique, avec parallélisme uniquement entre branches indépendantes et dans les budgets attribués.
6. Les progrès et previews sont publiés sur le bus ; l’annulation est contrôlée avant chaque opération et chaque étape de sampling.
7. Les images et métadonnées sont commitées atomiquement comme artefacts CAS versionnés.
8. Le job devient terminal, la réservation est libérée et les politiques keep-warm/éviction sont appliquées.

## 6. Modèle de sécurité

- Liaison par défaut à `127.0.0.1:8188`, sans auth ; un bind non-loopback sans auth produit une alerte bloquante configurable.
- API keys stockées sous forme de hash ; JWT signés avec secret local ; aucune télémétrie distante.
- Permissions plugin en liste blanche. Tous les chemins sont résolus puis vérifiés sous des racines autorisées.
- Réseau, subprocess, caméra et microphone refusés par défaut. Les permissions sont incluses dans le handshake du worker.
- Installations et téléchargements utilisent staging, hash, validation puis rename atomique.
- Les secrets de providers ne sont jamais renvoyés par l’API ni écrits dans les logs.

## 7. Gestion des erreurs et résilience

- Format d’erreur stable `{error: {code, message, details?, retryable?}}` avec codes HTTP cohérents.
- États de job monotones et transitions validées ; retry uniquement pour erreurs déclarées récupérables et avec plafond.
- Heartbeat et timeout différencient worker occupé, lent et mort. Un crash ferme les handles du propriétaire, journalise l’incident et ne touche pas aux autres workers.
- L’OOM suit : interruption du node, buffers temporaires, éviction pondérée, retry borné, offload si compatible, puis erreur claire.
- Les téléchargements sont repris via Range/ETag lorsque le provider le permet ; les partiels incompatibles sont supprimés.
- Les webhooks sont signés HMAC, persistés et retentés avec backoff exponentiel borné.

## 8. Contrats publics

- Préfixe immuable `/api/v1` et domaines listés dans la spécification.
- OpenAPI, Swagger et ReDoc toujours générés depuis les mêmes modèles Pydantic que les handlers.
- `Idempotency-Key` sur toutes les créations mutables concernées.
- WebSocket `/api/v1/events/ws`, SSE `/api/v1/events/sse`, métriques `/metrics`.
- Le catalogue de nodes expose identifiant/version, schémas d’entrées/sorties, ressources, permissions et cycle de vie sans importer l’implémentation.

## 9. Stratégie de tests

- Unitaires : types/capacités, compilation de graphe, priorités, budgets, manifests, CAS, providers, repositories et auth.
- Intégration : migrations, API complète, idempotence, WebSocket/SSE, worker spawn/heartbeat/crash/unload, annulation, retry, download/reprise et plugin isolé.
- E2E navigateur : création/édition/sauvegarde/validation d’un workflow, soumission, progression temps réel, annulation, navigation de toutes les surfaces.
- Runtime : tests contractuels sans Torch au boot, runtime déterministe, et tests GPU marqués séparément avec petit modèle local configuré par variable d’environnement.
- Performance : scripts reproductibles pour cold/warm start, RAM idle, import time, débit API/disque, VRAM et changement/éviction de modèle.
- Soak : scénarios 24 h / 1 000 jobs fournis et exécutables ; le CI court exécute une version réduite déterministe.

## 10. Critères de livraison

La livraison est acceptée lorsque les vingt critères V1 de `docs/SPECIFICATION.md` possèdent chacun un test, une sonde ou une procédure reproductible ; les tests non exécutables sur la machine macOS courante (CUDA/NVIDIA et soak 24 h) sont clairement marqués et accompagnés d’un runner Linux. Aucun placeholder fonctionnel, route factice ou bouton sans effet ne peut être présent dans la livraison.

## 11. Décisions explicites

- Python 3.14 est exigé à l’installation ; pas de baisse silencieuse de version.
- PyTorch est une dépendance d’extra runtime, jamais du core.
- SQLite est l’unique base V1 ; aucune abstraction distribuée prématurée.
- Le mode de démarrage est `forkserver` sous Linux et `spawn` sur macOS/Windows/CUDA lorsque nécessaire.
- Les modèles doivent être locaux avant une génération hors ligne ; les providers ne sont jamais consultés par le sampler.
- Les familles modernes futures entrent via un backend déclarant `ModelCapabilities`, sans ajouter d’options historiques globales.

