# Byte NDT Generic Array App V1

Application bilingue, indépendante du matériel, pour étudier un réseau 1D/2D à travers une interface fluide/solide plane.

## Fonctions

- dimensions, pitch, gap, nombre d'éléments et apodisation ;
- matériaux, onde P/S, fréquence et hauteur d'eau ;
- pilotage 3D par theta et phi ;
- pilotage seul ou focalisation finie ;
- champ vectoriel vx, vy, vz et module normalisé ;
- image linéaire ou dB avec contours −3, −6 et −12 dB ;
- étude fréquence–pitch ;
- export neutre CSV des lois focales ;
- export JSON de la configuration ;
- export CSV et PNG du champ.

## Architecture

```text
Moteur générique Byte NDT
        ↓
CSV + JSON neutres
        ↓
Plug-ins matériels vérifiés
        ├── Lecoeur US-ARRAY
        ├── Eddyfi Mantis / M2M
        ├── appareil de Shaun
        └── autres instruments
```

## Installation

```powershell
python -m pip install -r requirements.txt
```

## Test

```powershell
python -m pytest -q
```

## Lancement

```powershell
streamlit run app.py
```

## Validation

La chaîne reprend les équations MATLAB fournies. La validation numérique MATLAB/Python reste obligatoire avant tout usage matériel : comparaison des matrices de retard, des cartes de champ, de la position du maximum, de la largeur à −6 dB et de l'ordre des éléments.

## Étapes suivantes

- volumes 3D −3/−6/−12 dB ;
- parallélépipède orienté et descripteurs de tache ;
- interface courbe ;
- réponse sur indication ;
- plug-ins Lecoeur et Mantis ;
- génération de jeux de données pour machine learning.
