import numpy as np
import pyvista as pv

# --- Konstante ---
g = 1  # magnetische Monopolladung
epsilon = 1e-6  # zur Singularitätsvermeidung

# --- Gitter: 3D Volumen ---
x = np.linspace(-1, 1, 8)
y = np.linspace(-1, 1, 8)
z = np.linspace(-1, 1, 8)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

# --- Radius ---
R = np.sqrt(X**2 + Y**2 + Z**2)
R[R < epsilon] = epsilon  # Singularität vermeiden

# --- Monopolares Magnetfeld ---
factor = g / (4 * np.pi)
Bx = factor * X / R**3
By = factor * Y / R**3
Bz = factor * Z / R**3

# --- Punkte & Vektoren ---
points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
vectors = np.vstack((Bx.ravel(), By.ravel(), Bz.ravel())).T

# --- Richtung & Betrag ---
magnitudes = np.linalg.norm(vectors, axis=1)
directions = vectors / (magnitudes[:, np.newaxis] + epsilon)

# --- PolyData ---
pdata = pv.PolyData(points)
pdata["B"] = directions
pdata["magnitude"] = np.log10(magnitudes + epsilon)

# --- Glyphs ---
glyph_scale = 0.2
pdata.points -= 0.5 * directions * glyph_scale
glyphs = pdata.glyph(orient="B", scale=False, factor=glyph_scale)

# --- Plotter ---
plotter = pv.Plotter(window_size=(1000, 600))
plotter.set_background("white")
plotter.renderer.SetBackgroundAlpha(0)

plotter.add_mesh(glyphs, scalars="magnitude", cmap="plasma", show_scalar_bar=False)

# --- Kamera ---
plotter.view_vector([0, 7, .5], viewup=[0, 0, 1])
plotter.camera.zoom(1.4)
plotter.show_axes()

# --- Interaktiv + Kamera ausgeben ---
plotter.show(auto_close=False, interactive=True)

cpos = plotter.camera_position
print("View vector (camera position):", np.array(cpos[0]))
print("Focal point (looks at):       ", np.array(cpos[1]))
print("View up vector:               ", np.array(cpos[2]))

plotter.screenshot("dirac_B.png", transparent_background=True)
plotter.close()
