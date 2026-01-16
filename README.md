# Projet LSMC - Pricing d'Options Américaines (OpenMP)

Ce projet implémente la méthode de **Longstaff-Schwartz (LSMC)** pour l'évaluation d'options américaines (Put) par simulation de Monte Carlo et régression par moindres carrés.

Le cœur de calcul est écrit en **C++** (parallélisé avec **OpenMP**), et l'interface utilisateur est en **Python** (Tkinter) pour la configuration et la visualisation graphique.

## Architecture

Le projet est structuré comme suit :

- **`src/`** : Code source C++ (`main.cpp`, `lsmc.cpp`, `gbm.cpp`, etc.).
- **`include/`** : En-têtes (`.hpp`).
- **`ui.py`** : Interface graphique lanceur (Python).
- **`lsmc.sln` / `.vcxproj`** : Fichiers solution Visual Studio.

## Prérequis

### C++
- **Visual Studio 2022** (ou équivalent) avec le workload "Desktop development with C++".
- Support d'OpenMP (activé dans les propriétés du projet : `C/C++ -> Language -> OpenMP Support`).

### Python
- Python 3.x installé.
- Packages requis : `pandas`, `numpy`, `matplotlib`.
  ```bash
  pip install -r requirements.txt
  ```

## Compilation

1. Ouvrez `lsmc.sln` dans Visual Studio.
2. Sélectionnez la configuration **Release** (ou Debug) et la plateforme **x64**.
3. Faites un clic droit sur la solution -> **Recompiler** (Rebuild).
   *L'exécutable sera généré dans `x64/Release/lsmc.exe` ou `x64/Debug/lsmc.exe`.*

## Utilisation

Le moyen le plus simple est d'utiliser l'interface graphique :

1. Assurez-vous d'avoir compilé le projet.
2. Lancez le script Python :
   ```bash
   python ui.py
   ```
3. Dans l'interface :
   - Indiquez le chemin de l'exécutable (ex: `x64/Debug/lsmc.exe`).
   - Ajustez les paramètres (S0, K, T, Volatilité, Taux).
   - Choisissez le nombre de pas (N_steps) et de chemins (N_paths).
   - Cliquez sur **Lancer**.

Le programme C++ va générer les trajectoires, calculer le prix, et l'interface affichera les courbes et le résultat.

> **Note** : Le bouton "Lancer" nettoie automatiquement les anciens résultats avant chaque simulation pour garantir la fraîcheur des données.

## Auteurs

Projet réalisé par **Florian Barbe** (et Narjisse) dans le cadre du module **LSMC** (Centrale Nantes, 2025).
