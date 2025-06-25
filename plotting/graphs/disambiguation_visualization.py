import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
# Remove the face outline and only use eyes and mouth
# Eyes
eye1_x = np.array([-0.4])
eye1_y = np.array([0.5])
eye2_x = np.array([0.4])
eye2_y = np.array([0.5])

# Smile (parabola with smile)
smile_x = np.linspace(-0.5, 0.5, 9)
smile_y = 0.4 * (smile_x**2) - 0.3  # Smile instead of frown

# Combine eyes and mouth
X = np.vstack([
    np.stack([eye1_x, eye1_y], axis=1),
    np.stack([eye2_x, eye2_y], axis=1),
    np.stack([smile_x, smile_y], axis=1)
])

# Rotate 90 degrees clockwise
rotation_matrix = np.array([[0, 1], [-1, 0]])
X_rotated = X @ rotation_matrix.T

# Center the data
centroid = X_rotated.mean(axis=0)
X_centered = X_rotated - centroid

# Compute PCA
pca = PCA(n_components=2)
pca.fit(X_centered)
pc2 = pca.components_[1]  # Use second principal component

# Project points to second principal component
projections = X_centered @ pc2
projected_points = np.outer(projections, pc2)

# Distances from points to the PC line
dist_vectors = X_centered - projected_points
distances_sq = np.sum(dist_vectors**2, axis=1)

# Determine which side of the orthogonal line each point is on
orthogonal = np.array([-pc2[1], pc2[0]])  # rotate 90 degrees
sides = (X_centered @ pc2) > 0

# Sum squared distances for each side
sum_pos = distances_sq[sides].sum()
sum_neg = distances_sq[~sides].sum()

# Determine which side is +
sign_label = '+' if sum_pos > sum_neg else '-'
neg_label = '-' if sum_pos > sum_neg else '+'

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# Axis limits
xlim = (-0.75, 0.75)
ylim = (-0.75, 0.75)

for ax, show_distances in zip(axs, [False, True]):
    ax.scatter(X_centered[:, 0], X_centered[:, 1], s=40)
    ax.scatter([0], [0], color='black', zorder=5, label='Centroid')

    # Full PC2 line
    t = np.linspace(-2, 2, 100)
    pc2_line = np.outer(t, pc2)
    ax.plot(pc2_line[:, 0], pc2_line[:, 1], color='red', label="PC2")

    if show_distances:
        for i in range(len(X_centered)):
            ax.plot([X_centered[i, 0], projected_points[i, 0]],
                    [X_centered[i, 1], projected_points[i, 1]],
                    color='gray', linewidth=1)

        # Full orthogonal line (purple, solid)
        ortho_line = np.outer(t, orthogonal)
        ax.plot(ortho_line[:, 0], ortho_line[:, 1], color='purple', linewidth=1.5)

        # Add +/- sign on the PC2 line
        offset = 0.42
        ax.text(offset * pc2[0], offset * pc2[1] + 0.2, sign_label,
                fontsize=50, ha='center', va='center', color='black')

        ax.text(offset * pc2[0], offset * pc2[1] + 0.1, f"(val: {sum_pos:.2f})",
                fontsize=10, ha='center', va='center', color='black')

        ax.text(-offset * pc2[0], -offset * pc2[1] + 0.2, neg_label,
                fontsize=30, ha='center', va='center', color='black')

        ax.text(-offset * pc2[0], -offset * pc2[1] + 0.1, f"(val: {sum_neg:.2f})",
                fontsize=10, ha='center', va='center', color='black')

    ax.set_aspect('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title("With Sum of Squared Residuals Measure" if show_distances else "Original")
    ax.legend()

plt.tight_layout()
plt.savefig("disambiguation_graphic.pdf")
plt.show()
