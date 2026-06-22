# Plateforme Abeille 

## Commandes à retenir
Pour copier les fichiers de Windows à Linux (dans le terminal Linux!) :

```bash
cp -r /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/Gazebo_defs/* ~/PX4-Autopilot/BEE_LAND/
```

cp -r /mnt/c/Users/Pipef/OneDrive/Academiques/Stage/CodeGit/Python_comm/*.py ~/PX4-Autopilot/BEE_LAND/controller/

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
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/PX4-Autopilot/build/px4_sitl_default/src/modules/simulation/gz_plugins:$GZ_SIM_SYSTEM_PLUGIN_PATH
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
PX4_GZ_MODEL_POSE="0,0,2.4,0,0,0" \  ## (x, y, z, roll, pitch, yaw)
PX4_GZ_WORLD=bee_platform \
make px4_sitl gz_x500
```
Une fois la simulation marche, il faut configurer les paramètres du quadricopteur : 

```bash
param set SYS_HAS_BARO 1
param set SYS_HAS_MAG 1

param set EKF2_BARO_CTRL 1
param set EKF2_HGT_REF 0

param set EKF2_MAG_TYPE 1
param set EKF2_MAG_CHECK 0
param set COM_ARM_MAG_STR 0

param set EKF2_OF_CTRL 0
param set EKF2_RNG_CTRL 0

param set COM_RC_IN_MODE 4
param set COM_RCL_EXCEPT 31
param set NAV_RCL_ACT 0

param set NAV_DLL_ACT 0
param set COM_DLL_EXCEPT 0

param set COM_ARM_WO_GPS 1
param set COM_ARM_ODID 0

param set COM_CPU_MAX -1
param set COM_RAM_MAX -1
param set COM_POWER_COUNT 0
param set CBRK_SUPPLY_CHK 894281
param set CBRK_USB_CHK 197848

param save
```

SYS_HAS_BARO, SYS_HAS_MAG: keep simulated barometer/magnetometer enabled.
EKF2_*: use baro/GPS/mag cleanly for this early SITL setup; disable optical-flow/range fusion until those sensors exist.
COM_RC_IN_MODE, COM_RCL_EXCEPT, NAV_RCL_ACT: allow operation without RC/manual input.
NAV_DLL_ACT, COM_DLL_EXCEPT: allow operation without QGroundControl.
COM_ARM_*: relax GPS/OpenDroneID/mag arming blockers for simulation.
COM_CPU_MAX, COM_RAM_MAX, COM_POWER_COUNT, CBRK_*: remove SITL/WSL system-power health blockers.

## Modification des oscillations
Pour modifier les frequences (en Hz) et les amplitudes des oscillations (en m), il faut ouvrir le 'world' bee_platform.sdf :

```xml
<x_amplitude>1.0</x_amplitude>
<x_frequency>0.10</x_frequency>

<z_amplitude>0.30</z_amplitude>
<z_frequency>0.20</z_frequency>
```

## Inclusion du modèle-abeille
Le modèle utilisé est une modification du x500, modèle très connu et utilisé. Sa modification nous permets d'ajouter des senseurs (comme une caméra) pour estimer le flux optique. 

```bash
PX4_SYS_AUTOSTART=4001 \
PX4_SIMULATOR=gz \
PX4_SIM_MODEL=bee_x500 \
PX4_GZ_MODEL_POSE="0,0,2.4,0,0,0" \
PX4_GZ_WORLD=bee_platform \
./build/px4_sitl_default/bin/px4
```