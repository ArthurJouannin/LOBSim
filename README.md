# LOBSim: Simulateur de Carnet d'Ordres Haute Fréquence

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/) [![ArXiv](https://img.shields.io/badge/arXiv-2312.08927v5-red)](https://arxiv.org/abs/2312.08927)

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

<img width="949" alt="Capture d'écran 2025-05-24 221720" src="https://github.com/user-attachments/assets/2c6b91ee-ff55-4539-80b6-b8b3fa36bd85" />
