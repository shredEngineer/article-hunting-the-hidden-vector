import numpy as np
import pyvista as pv

# --- Konstante ---
g = 1  # magnetische Monopolladung (Skalierung)
epsilon = 1e-6  # für Singularitätsvermeidung

# --- Gitter: 3D Volumen ---
x = np.linspace(-1, 1, 10)
y = np.linspace(-1, 1, 10)
z = np.linspace(-1, 1, 3)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# --- Betrag r und r - z ---
R = np.sqrt(X**2 + Y**2 + Z**2)
R_minus_z = R - Z

# --- Singularitäten vermeiden ---
R[R < epsilon] = epsilon
R_minus_z[np.abs(R_minus_z) < epsilon] = epsilon
denom = R * R_minus_z
denom[denom < epsilon] = epsilon

# --- Dirac-Vektorpotential ---
factor = g / (4 * np.pi)
Ax = factor * (-Y) / denom
Ay = factor * X / denom
Az = np.zeros_like(Ax)

# --- Punkte & Vektoren ---
points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
vectors = np.vstack((Ax.ravel(), Ay.ravel(), Az.ravel())).T

# --- Richtung & Betrag ---
magnitudes = np.linalg.norm(vectors, axis=1)
directions = vectors / (magnitudes[:, np.newaxis] + epsilon)

# --- PolyData ---
pdata = pv.PolyData(points)
pdata["A"] = directions
pdata["magnitude"] = np.log10(magnitudes + epsilon)

# --- Glyphs ---
glyph_scale = 0.2
pdata.points -= 0.5 * directions * glyph_scale
glyphs = pdata.glyph(orient="A", scale=False, factor=glyph_scale)

# --- Plotter ---
plotter = pv.Plotter(window_size=(1000, 600))
plotter.set_background("white")
plotter.renderer.SetBackgroundAlpha(0)

plotter.add_mesh(glyphs, scalars="magnitude", cmap="viridis", show_scalar_bar=False)

# --- Dirac-String als roter Zylinder (nur von z=0 nach z=-zl) ---
zl = 1.6
string_center = (0, 0, -zl/2)
string_height = zl
string_radius = 0.02
dirac_string = pv.Cylinder(center=string_center, direction=(0, 0, -1),
                           radius=string_radius, height=string_height)
plotter.add_mesh(dirac_string, color="red", opacity=0.6)

# --- Pfeilspitze am Ende des Dirac-Strings (bei z = -zl) ---
arrow_tip = pv.Cone(center=(0, 0, -zl), direction=(0, 0, -1),
                    height=0.2, radius=0.05)
plotter.add_mesh(arrow_tip, color="red", opacity=0.8)

# --- Beschriftung der z-Ebenen ---
plotter.add_point_labels([ (-1.5, .4, -1), (-1.5, .4, 0), (-1.5, .4, 1) ],
                         ["z = –1", "z = 0", "z = 1"],
                         font_size=32, text_color="black", point_color="white",
                         shape_opacity=0.0, always_visible=True)

# --- Kamera ---
plotter.view_vector([0, 8, 2], viewup=[0, 0, 1])
plotter.camera.zoom(1.4)
plotter.show_axes()

# --- Screenshot ---
plotter.show(auto_close=False, interactive=False)

cpos = plotter.camera_position
print("View vector (camera position):", np.array(cpos[0]))
print("Focal point (looks at):       ", np.array(cpos[1]))
print("View up vector:               ", np.array(cpos[2]))

plotter.screenshot("dirac_A.png", transparent_background=True)
plotter.close()
