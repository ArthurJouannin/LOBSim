# 1: LOBSim: Simulateur de Carnet d'Ordres Haute Fréquence

## Description

LOBSim est un simulateur de carnet d'ordres (Limit Order Book) basé sur un Processus de Hawkes Composé. Il modélise la dynamique des marchés financiers haute fréquence en simulant 12 types d'événements de trading, calibrés avec une méthode inspirée de Kirchner (2017) et Bacry et al. (2016). 

Ce projet s'appuie sur l'article académique « Limit Order Book Dynamics and Order Size Modelling Using Compound Hawkes Process » (Konark Jain *et al.*, arXiv:2312.08927v5).

## Fonctionnalités principales

* **12 types d'événements** : Limit Orders (LO), Market Orders (MO), Cancel Orders (CO) sur trois niveaux de prix (ask\_p1, ask0, ask\_m1, bid\_p1, bid0, bid\_m1).
* **Calibration automatique** : paramètres μ, noyaux power-law, facteur temporel intra-journalier.
* **Visualisations intégrées** :
  
  * Évolution du mid-price
  * Évolution du spread (en ticks)
  * Répartition des événements par type
  * Distribution des spreads

## Installation et Pré-requis

```bash
# Cloner le dépôt
git clone https://github.com/Tuturj/LOBSim.git

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # sous Linux/Mac
venv\Scripts\activate   # sous Windows

# Installer les dépendances
pip install numpy pandas matplotlib
```

**Dépendances** : numpy, pandas, matplotlib

## Méthodologie Scientifique

Pour plus de détails, voir l’article : [arXiv:2312.08927v5](https://arxiv.org/abs/2312.08927).

## Résultats

![Result1](https://github.com/user-attachments/assets/f5926498-0689-4974-8600-f1786e226e3d)

# 2: Market impact d'Ordres market exogène

## Description

On choisit d'ajouter la possibilité d'executer des ordres exogènes au cours de la simulation. Ces ordres sont executés sous forme de méta-ordres linéaires de la taille la plus grande possible. Autrement dit, une fois executé, l'ordre absorbera toute la liquidité régulièrement jusqu'à ce que l'ordre soit entièrement filled. Un plot du prix et du market impact est obtenu.
Une des possibilités pour une implémentation future est de mesurer la différence de coût et d'impact entre cette méthode et le modèle d'optimisation optimal d'Almgren-Chriss.
LOBSim_MI.py : 
- Simule dynamiquement un carnet d’ordres (Limit Order Book, LOB) avec différents types d’événements (orders limit, market et cancel).
- Gère et exécute des meta‐ordres découpés en child orders selon un planning linéaire.
- Calcule l’impact de marché (immédiat, temporaire, permanent) lié à l’exécution de ces meta‐ordres.
- Produit des graphiques de l’évolution du prix mid‐quote et de la décomposition d’impact.

## Installation

1. Cloner le dépôt :
    ```bash
    git clone https://github.com/Tuturj/LOBSim.git
    ```
2. Créer et activer un environnement virtuel (optionnel mais recommandé) :
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # sous macOS/Linux
    venv\Scripts\activate     # sous Windows
    ```
3. Installer les dépendances :
    ```bash
    pip install numpy, matplotlib
    ```
    
## Résultats


