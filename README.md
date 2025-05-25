# LOBSim: Simulateur de Carnet d'Ordres Haute Fréquence

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
git clone https://github.com/votre-utilisateur/LOBSim.git
cd LOBSim

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
