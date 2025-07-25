import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import binary_dilation, generate_binary_structure

# Function to generate the local scope mask
def local_scope_mask(block_mask: np.ndarray, scope_only: bool = False):
    struct = generate_binary_structure(rank=block_mask.ndim, connectivity=1)
    expanded_mask = binary_dilation(block_mask, structure=struct)
    return expanded_mask & ~block_mask if scope_only else expanded_mask

# Create 5x5x5 array with a 2x2x2 block in the center
arr = np.zeros((5, 5, 5), dtype=int)
arr[2:4, 2:4, 2:4] = 1

# Generate local scope mask (block + 6 face neighbors)
mask = local_scope_mask(arr == 1)

# Prepare data for plotting
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')

# Plot voxels
# x, y, z = np.where(mask)
# colors = np.empty(mask.shape, dtype=object)
# colors[arr == 1] = 'red'      # original block
# colors[(mask) & (arr == 0)] = 'blue'  # face neighbors
#
# ax.voxels(mask, facecolors=colors, edgecolor='k')
# ax.set_title("3D Local Scope Around a Cube (Red = Block, Blue = Scope)")
# plt.show()

#########


# Re-plot the 3D visualization without axes for a cleaner look

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot block (opaque red)
ax.voxels(arr == 1, facecolors='red', edgecolor='k', alpha=1.0)

# Plot scope (translucent blue)
scope_only = (mask) & (arr == 0)
ax.voxels(scope_only, facecolors='blue', edgecolor='k', alpha=0.4)

# Remove axes and grid
ax.set_axis_off()

plt.tight_layout()
plt.show()