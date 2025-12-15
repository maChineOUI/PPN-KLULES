
# **KLULES : Portage Kokkos 5.0 de la proxy-app LULESH (version CPU hôte)**

* **Encadrant** : Gabriel DOS SANTOS
* **Contact** : [gabriel.dos-santos@uvsq.fr](mailto:gabriel.dos-santos@uvsq.fr)

La proxy-application **L L U L E S H** (*Livermore Unstructured Lagrangian Explicit Shock Hydrodynamics*) est un mini-code d’hydrodynamique développé au LLNL. Elle résout un problème de type Sedov sur un maillage hexaédrique non structuré, avec de nombreux accès indirects (connectivité éléments–nœuds).

Le projet **K L U L E S** (Kokkos Learning Unstructured Lagrangian Explicit Shocks) vise à porter cette application vers **Kokkos 5.0**, afin de préparer un code de référence pédagogique, portable sur différentes architectures mais dans cette première étape limité au CPU hôte.

---

# **1. Objectifs du projet et principaux changements liés à Kokkos**

## **Objectifs pédagogiques (S1)**

* Comprendre l’implémentation de référence LULESH (C++ séquentiel / OpenMP).
* Représenter les données de maillage et les champs physiques avec `Kokkos::View`.
* Remplacer les boucles séquentielles par des boucles parallèles Kokkos (`parallel_for`, `parallel_reduce`).
* Valider la correction numérique (comparaison pression / énergie avec la version référence).
* Préparer la parallélisation future (OpenMP / GPU).

## **Principaux aspects du portage Kokkos**

Dans cette version :

* Remplacement de tableaux bruts par des `Kokkos::View`.
* Remaniement des boucles :

  ```cpp
  for (Index_t i = 0; i < numElem; ++i)
  ```

  devient

  ```cpp
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecSpace>(0, numElem),
                       KOKKOS_LAMBDA (const Index_t i) { ... });
  ```
* Utilisation de `parallel_reduce` pour les contraintes de temps (Courant & Hydro).
* Encapsulation des données du domaine dans des structures compatibles Kokkos.
* Utilisation d’un *execution space* CPU (`Serial` ou `DefaultExecutionSpace` configuré hôte).

---

# **2. Structure du projet**

Arborescence actuelle :

```text
├── build
├── build.log
├── CMakeLists.txt
├── lulesh_tuple.h
├── lulesh-comm.cc
├── lulesh-init.cc
├── lulesh-util.cc
├── lulesh-viz.cc
├── lulesh.cc
└── lulesh.h
```

---

## **2.1 Fichiers portés / développés**

### **lulesh.cc — point d’entrée**

* parsing des arguments,
* initialisation Kokkos,
* création du domaine (Views + données),
* boucle en temps : kernels nodaux, élémentaires, calcul de `dt`.

---

### **lulesh-init.cc — initialisation**

* maillage cartésien 3D,
* connectivité éléments–nœuds,
* initialisation des champs physiques : densité, énergie, pression, Sedov,
* allocation et `deep_copy` des Views Kokkos.

---

### **lulesh-util.cc — noyaux numériques**

Contient les versions Kokkos des kernels :

* **LagrangeNodal** : forces, accélérations, vitesses, positions ;
* **LagrangeElements** : volumes, densités, énergie interne, pression, viscosité artificielle；
* contraintes Courant & hydro via `parallel_reduce`；
* utilitaires de réduction (min/max/énergie globale).

---

### **lulesh.h — structures de données**

* Déclare les `Kokkos::View` du domaine.
* Prototypes des fonctions.
* Constantes numériques (coefficients, tailles, seuils physiques).

---

## **2.2 Différences avec une version complète Kokkos (CPU + GPU)**

### **Non inclus :**

* visualisation (`lulesh-viz.cc`),
* code GPU (TeamPolicy, UVM...),
* MPI (multi-domaines),
* configuration complexe multi-backend.

### **Inclus :**

* un seul exécutable hôte ;
* Makefile minimal ;
* dépendance unique : Kokkos + STL.

---

# **3. Pistes d’évolution**

## **Support multi-backend**

* Paramétrage `ExecutionSpace` pour OpenMP / CUDA / SYCL.
* Options CMake : `Kokkos_ENABLE_OPENMP`, `Kokkos_ENABLE_CUDA`, etc.

## **Retour MPI**

* Décomposition 3D, SBN, échanges de nœuds fantômes.
* Exécution hybride MPI + Kokkos.

## **Visualisation I/O scientifique**

* Export VTK / SILO pour Paraview / VisIt.
* Dump périodique de l’onde de choc Sedov.

## **Tests & Benchmark**

* Tests unitaires sur kernels isolés.
* Benchmark comparatif :

  * C++ séquentiel,
  * Kokkos Serial / OpenMP,
  * futur GPU.

## **Nettoyage / factorisation**

* Policies Kokkos réutilisables.
* Documentation renforcée.

---

# **4. Organisation du travail**

## **Membres du groupe**

ELARAR Haitam (GitHub : HaitamELARAR)

Huang Yupan (GitHub : HYupan)

Fang Zijie (GitHub : FANG-leo)

Wei Wei (GitHub : maChineOUI)

## **Répartition (au 14 décembre)**

### **Huang, Fang, Wei：**

* étude LULESH original,
* portage Kokkos : `lulesh.cc`, `lulesh-init.cc`, `lulesh-util.cc`, `lulesh.h`.

### **ELARAR：**

* vérification de la cohérence numérique,
* relecture, tests, débogage,
* préparation des scénarios d’exécution et profilage.

---
