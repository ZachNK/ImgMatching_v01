import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from pathlib import Path

OUTPUT_DIR = Path("/exports/figs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
grid_path = Path(r"/exports/dinov3_embeds/jamshill_flight/vith16+/251124155712/PatchGrid/PatchGrid_res1024_raw_dinov3_vith16plus_LVD_251124155712_00092.npy")
img_path  = Path(r"/opt/datasets/01_01_jamshill_data_flight/251124155712/251124155712_00092.jpg")

grid = np.load(grid_path)  # (H, W, C)
H, W, C = grid.shape

mode = "norm"      # "norm" or "channel"
channel = 0

if mode == "norm":
    z = np.linalg.norm(grid, axis=-1)  # (H, W)
else:
    z = grid[..., channel]             # (H, W)

x, y = np.meshgrid(np.arange(W), np.arange(H))
x = x.ravel(); y = y.ravel(); z = z.ravel()

fig = plt.figure(figsize=(12, 5))

# 왼쪽: 3D scatter (z=토큰값)
ax = fig.add_subplot(1, 2, 1, projection="3d")
sc = ax.scatter(x, y, z, c=z, cmap=cm.magma, s=4)
ax.set_title(f"PatchGrid 3D ({mode})\n{grid_path.name}")
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("value")
fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)

# 오른쪽: 원본 이미지
ax2 = fig.add_subplot(1, 2, 2)
img = Image.open(img_path).convert("RGB")
ax2.imshow(img)
ax2.set_title(f"Original Image\n{img_path.name}")
ax2.axis("off")

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "patchgrid_with_image.png", dpi=300, bbox_inches="tight")
plt.show()
