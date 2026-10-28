import json
from pathlib import Path
import numpy as np

# =========================
# Paramètres
# =========================

output_file = Path("output/receivers_generated.json")

dx = 1.0
half_width = 20.0

offsets = np.arange(-half_width, half_width + dx, dx)

# =========================
# Receivers ST
# =========================

st_radius = 10.0      # rayon du cercle [m]
n_st_circle = 10      # nombre de points autour de chaque ST

st_receivers = {
    "STA": [4573.68, -120.98],
    "STB": [16674.96, -159.86],
}

def generate_ellipse_receivers(
    receiver_dict,
    radius_x=10.0,
    radius_z=1.0,
    n_points=12
):
    names = []
    positions = []

    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    for base_name, (x0, z0) in receiver_dict.items():

        # receiver réel
        names.append(f"{base_name}_center")
        positions.append([round(x0, 3), round(z0, 3)])

        # ellipse autour
        for i, theta in enumerate(angles):

            x = x0 + radius_x * np.cos(theta)
            z = z0 + radius_z * np.sin(theta)

            names.append(f"{base_name}_c{i:02d}")
            positions.append([
                round(x, 3),
                round(z, 3)
            ])

    return names, positions

# =========================
# Receivers OBS
# =========================

obs_receivers = {
    "OBS1": [3039.6, -106.84],
    "OBS2": [4818.48, -149.81],
    "OBS3": [6960.48, -267.20],
    "OBS4": [9196.32, -232.49],
    "OBS5": [11938.08, -261.11],
    "OBS6": [14320.8, -249.19],
    "OBS7": [16993.2, -179.88],
    "OBS8": [19567.68, -197.75],
}

# =========================
# Fonction génération
# =========================

def generate_receivers(receiver_dict):
    names = []
    positions = []

    for base_name, (x0, z0) in receiver_dict.items():

        for offset in offsets:

            x = x0 + offset
            z = z0

            suffix = int(offset)

            if suffix < 0:
                name = f"{base_name}_m{abs(suffix):02d}"

            elif suffix > 0:
                name = f"{base_name}_p{suffix:02d}"

            else:
                name = f"{base_name}_000"

            names.append(name)
            positions.append([
                round(x, 3),
                round(z, 3)
            ])

    return names, positions

# =========================
# Génération
# =========================

st_names, st_positions = generate_ellipse_receivers(
    st_receivers,
    radius_x=10.0,
    radius_z=1.0,
    n_points=20
)

obs_names, obs_positions = generate_receivers(obs_receivers)

# =========================
# JSON final
# =========================

receivers = {
    "acoustic_2d": {
        "name": st_names,
        "position": st_positions,
        "path": "receivers_acoustic.csv"
    },

    "viscoelastic_2d": {
        "name": obs_names,
        "position": obs_positions,
        "path": "receivers_viscoelastic.csv"
    }
}

with open(output_file, "w") as f:
    json.dump(receivers, f, indent=2)

print(f"Fichier créé : {output_file}")

print(f"Nombre ST : {len(st_names)}")
print(f"Nombre OBS : {len(obs_names)}")
print(f"Total : {len(st_names) + len(obs_names)}")