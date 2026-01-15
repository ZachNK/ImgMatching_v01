import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from pathlib import Path

OUTPUT_DIR = Path("/exports/figs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
grid_path = Path(r"/exports/dinov3_embeds/jamshill_flight/vith16+/251124155712/PatchGrid/PatchGrid_res1024_raw_dinov3_vith16plus_LVD_251124155712_00092.npy")
img_path  = Path(r"/opt/datasets/01_01_jamshill_data_flight/251124155712/251124155712_00092.jpg")

grid = np.load(grid_path)
H, W, C = grid.shape

mode = "channel"   # "norm" or "channel"
channel = 0

z = np.linalg.norm(grid, axis=-1) if mode == "norm" else grid[..., channel]
x, y = np.meshgrid(np.arange(W), np.arange(H))

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "xy"}]],
    column_widths=[0.6, 0.4],
    subplot_titles=[
        f"PatchGrid Surface ({mode})<br>{grid_path.name}",
        f"Original Image<br>{img_path.name}",
    ]
)

fig.add_trace(
    go.Surface(z=z, x=x, y=y, colorscale="Viridis"),
    row=1, col=1
)

img = Image.open(img_path).convert("RGB")
fig.add_trace(go.Image(z=np.array(img)), row=1, col=2)

fig.update_layout(
    title=dict(text="PatchGrid 3D Surface + Image<br>", x =0.5, xanchor="center"),
    height=800,
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="value")
)

# HTML 저장
save_path = OUTPUT_DIR / "patchgrid_surface_with_image.html"
fig.write_html(save_path)
fig.show()
