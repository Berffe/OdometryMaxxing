# Plateforme Abeille 

## Commandes à retenir
Pour copier les fichiers de Windows à Linux (dans le terminal Linux!) :

```bash
cp -r /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/Gazebo_defs/* ~/PX4-Autopilot/BEE_LAND/
```
Le contraire, Linux à Windows (dans le terminal Windows!) :

```bash
scp -r ~/PX4-Autopilot/BEE_LAND/plugins/oscillating_platform_controller C:\Users\Pipef\OneDrive\Academiques\Stage\CodeGit\Gazebo_defs\*
```
## Définition de la plateforme
On définit la plateforme dans un fichier centralisé (BEE_LAND), mais il faut le synchroniser aux fichiers internes de l'environnement PX4 :

```text
~/PX4-Autopilot/BEE_LAND/
├── worlds/
│   └── bee_platform.sdf              # définition du 'world'
├── plugins/
│   └── oscillating_platform_controller/   # contrôleur de la plateforme (oscillatoire)
└── README.md                         # quelques justifications
```

Ainsi, la synchronisation se fait par :

```bash
ln -s ~/PX4-Autopilot/BEE_LAND/worlds/bee_platform.sdf  ~/PX4-Autopilot/Tools/simulation/gz/worlds/bee_platform.sdf

ln -s ~/PX4-Autopilot/BEE_LAND/plugins/oscillating_platform_controller \
      ~/PX4-Autopilot/src/modules/simulation/gz_plugins/oscillating_platform_controller
```

Le plugin ci-dessous a été construit :

```text
libOscillatingPlatformController.so
custom::OscillatingPlatformController
```

Et le  `bee_platform.sdf` world lance ce plugin directement.

De plus, il a fallu modifier le fichier qui définit les plugins disponibles pour le projet. Dans le fichier :

```bash
src/modules/simulation/gz_plugins/CMakeLists.txt
```

on ajoute juste en dessous de toutes les autres inclusions de 'subdirectory' : 

```cmake
add_subdirectory(oscillating_platform_controller)
```

Finalement, on relance le 'build' de l'environnement :

```bash
cd ~/PX4-Autopilot
make px4_sitl
```

Une petite vérification peut aussi être fait : 

```bash
find build/px4_sitl_default -name 'libOscillatingPlatformController.so'
```

## Lancement de la plateforme 
Plateforme seule :

```bash
gz sim Tools/simulation/gz/worlds/bee_platform.sdf
```

Plateforme avec le modèle du drone (en lui positionant ou vous voulez) : 

```bash
PX4_GZ_MODEL_POSE="0,0,4,0,0,0" \  ## (x, y, z, roll, pitch, yaw)
PX4_GZ_WORLD=bee_platform \
make px4_sitl gz_x500
```

## Modification des oscillations
Pour modifier les frequences (en Hz) et les amplitudes des oscillations (en m), il faut ouvrir le 'world' bee_platform.sdf :

```xml
<x_amplitude>1.0</x_amplitude>
<x_frequency>0.10</x_frequency>

<z_amplitude>0.30</z_amplitude>
<z_frequency>0.20</z_frequency>
```