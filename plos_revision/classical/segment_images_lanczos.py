"""Segment image using CPU weighted graphs and Lanczos Ncut.

Images in INPUT_DIR are processed independently. For each image and parameter
combination, the script saves a color visualization and a run-length .seg file.
The matrix-vector product in Lanczos uses SciPy sparse multiplication on the CPU.
"""

from datetime import datetime
import os

import scipy.sparse as sp
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from skimage import color, io

INPUT_DIR = "image_test"
OUTPUT_DIR = "image_result"
SIGMA_I_VALUES = (0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.013, 0.014, 0.016, 0.017, 0.019, 0.02, 0.023, 0.024, 0.027, 0.029, 0.03, 0.032, 0.035, 0.037, 0.038, 0.041, 0.045, 0.049, 0.052, 0.055, 0.057, 0.059, 0.06, 0.063, 0.065, 0.069, 0.071, 0.074, 0.077, 0.078, 0.08, 0.082, 0.084, 0.087, 0.089, 0.093, 0.096, 0.099, 0.11, 0.12, 0.13, 0.15, 0.16, 0.18, 0.19, 0.21, 0.24,0.27, 0.29, 0.3)
# SIGMA_I_VALUES = (0.009,)
SIGMA_X_VALUES = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
# SIGMA_X_VALUES = (8,)
K_NEIGHBORS = 504


def compute_weight_matrix_coo_knn_cpu(image, sigma_i, sigma_x, k_neighbors=K_NEIGHBORS):
    """Build a COO weight matrix using spatial kNN and CPU affinity values."""
    h, w, c = image.shape
    coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T
    features = image.reshape(-1, c)

    knn = NearestNeighbors(n_neighbors=k_neighbors, algorithm="ball_tree").fit(coords)
    distances, indices = knn.kneighbors(coords)

    row_idx = np.asarray(np.repeat(np.arange(len(coords)), k_neighbors), dtype=np.int32)
    col_idx = np.asarray(indices.flatten(), dtype=np.int32)
    features_cpu = np.asarray(features, dtype=np.float32)
    distances_cpu = np.asarray(distances, dtype=np.float32)
    values = np.zeros(len(row_idx), dtype=np.float32)

    for i in range(len(coords)):
        neighbor_idx = indices[i]
        diff_feature = features_cpu[i] - features_cpu[neighbor_idx]
        feature_weight = np.exp(-np.sum(diff_feature ** 2, axis=1) / (2 * sigma_i ** 2))
        spatial_weight = np.exp(-distances_cpu[i] ** 2 / (2 * sigma_x ** 2))
        values[i * k_neighbors:(i + 1) * k_neighbors] = feature_weight * spatial_weight

    directed_weights = sp.coo_matrix(
        (values.ravel(), (row_idx.ravel(), col_idx.ravel())),
        shape=(len(coords), len(coords)),
    )
    # Spatial kNN can be directed. Average reciprocal weights so that the
    # normalized graph Laplacian is symmetric, as required by this Lanczos step.
    weights_csr = directed_weights.tocsr()
    symmetric_weights = (weights_csr + weights_csr.T) * np.float32(0.5)
    return symmetric_weights.tocoo()


def compute_laplacian_coo(W_coo):
    """Return the unnormalized graph Laplacian and degree matrix."""
    degrees = np.asarray(W_coo.sum(axis=1)).flatten()
    D_coo = sp.coo_matrix(
        (degrees, (np.arange(len(degrees)), np.arange(len(degrees)))),
        shape=W_coo.shape,
    )
    return D_coo - W_coo, D_coo


def compute_ncut_lanczos(W_coo, k=2, max_iter=1000, tol=1e-5):
    """Approximate the smallest nonzero eigenvectors of I - D^-1/2 W D^-1/2.

    Use an explicit Lanczos iteration and SciPy sparse matrix-vector
    multiplication rather than a library eigensolver.
    """
    n_vertices = W_coo.shape[0]
    degrees = np.asarray(W_coo.sum(axis=1)).flatten()
    inv_sqrt_degree = 1.0 / np.sqrt(degrees + 1e-8)
    row, col = W_coo.row, W_coo.col
    data = W_coo.data * inv_sqrt_degree[row] * inv_sqrt_degree[col]
    W_norm = sp.coo_matrix((data, (row, col)), shape=W_coo.shape)

    def A_mul(x):
        """Apply the Lanczos operator using dense matrix-vector multiplication."""
        Wx = W_norm.toarray() @ x
        return x - Wx

    Q, alphas, betas = [], [], []
    q = np.random.randn(n_vertices).astype(np.float32)
    q /= np.linalg.norm(q)
    Q.append(q)
    beta = 0.0
    q_prev = np.zeros_like(q)

    for _ in range(max_iter):
        z = A_mul(Q[-1])
        alpha = np.dot(Q[-1], z)
        alphas.append(alpha)
        z = z - alpha * Q[-1] - beta * q_prev

        # Reorthogonalize against the previously computed Lanczos vectors.
        for q_i in Q:
            z -= np.dot(q_i, z) * q_i

        beta = np.linalg.norm(z)
        if beta < tol or len(alphas) >= k + 50:
            break
        betas.append(beta)
        q_prev = Q[-1]
        Q.append(z / beta)

    m = len(alphas)
    T = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        T[i, i] = alphas[i]
        if i > 0:
            T[i, i - 1] = T[i - 1, i] = betas[i - 1]

    vals, vecs = np.linalg.eigh(T)
    sorted_idx = np.argsort(vals)
    nonzero = np.where(np.abs(vals[sorted_idx]) > 1e-5)[0]
    vecs = vecs[:, sorted_idx][:, nonzero[:k]]
    Q_mat = np.stack(Q, axis=1)
    return Q_mat @ vecs


def assign_labels(eigenvectors, k):
    """Cluster the spectral embedding into k segments."""
    return KMeans(n_clusters=k, random_state=0).fit(eigenvectors).labels_


def save_segmentation(image, labels, k, output_path):
    """Save a visualization using the mean image color for each segment."""
    h, w, _ = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(k):
        mask = labels.reshape(h, w) == i
        cluster_pixels = image[mask]
        mean_color = (
            (cluster_pixels.mean(axis=0) * 255).astype(np.uint8)
            if len(cluster_pixels) > 0 else np.array([0, 0, 0], dtype=np.uint8)
        )
        segmented_image[mask] = mean_color
    io.imsave(output_path, segmented_image)


def save_seg_file(labels, image_shape, output_path, image_name="image"):
    """Save run-length SEG data with inclusive start and end columns."""
    h, w = image_shape[:2]
    segments = len(np.unique(labels))
    header = [
        "format ascii cr",
        f"date {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}",
        f"image {image_name}",
        "user 1102",  # Preserve the user field from the original SEG header.
        f"width {w}",
        f"height {h}",
        f"segments {segments}",
        "gray 0",
        "invert 0",
        "flipflop 0",
        "data",
    ]
    data_lines = []
    for row in range(h):
        row_labels = labels[row, :]
        start_col = 0
        current_label = row_labels[0]
        for col in range(1, w):
            if row_labels[col] != current_label:
                # The last column of the previous run is col - 1.
                data_lines.append(f"{current_label} {row} {start_col} {col - 1}")
                start_col = col
                current_label = row_labels[col]
        data_lines.append(f"{current_label} {row} {start_col} {w - 1}")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(header) + "\n")
        file.write("\n".join(data_lines) + "\n")
    print(f"SEG file saved: {output_path}")


def normalized_cuts_lanczos(image_name, image_path, output_path, k, sigma_i, sigma_x):
    """Segment one image patch and save its visualization and SEG labels."""
    image = io.imread(image_path)
    image = color.gray2rgb(image) if image.ndim == 2 else image[:, :, :3] if image.shape[2] == 4 else image
    image = image / 255.0
    W_coo = compute_weight_matrix_coo_knn_cpu(image, sigma_i, sigma_x)
    vecs = compute_ncut_lanczos(W_coo, k)
    labels = assign_labels(vecs, k)
    save_segmentation(image, labels, k, output_path + ".jpg")
    save_seg_file(labels.reshape(image.shape[:2]), image.shape, output_path + ".seg", image_name)
    del W_coo, vecs


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"Input directory does not exist: {INPUT_DIR}")
        return
    image_files = sorted(
        filename for filename in os.listdir(INPUT_DIR)
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    )
    if not image_files:
        print(f"No image files found in {INPUT_DIR}")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for index, filename in enumerate(image_files, start=1):
        # Extract the number of segments k from the end of the filename, or specify it directly for the image to be segmented.
        k = int(os.path.splitext(filename)[0].split("_")[-1].lstrip("Kk"))
        image_path = os.path.join(INPUT_DIR, filename)
        print(f"Processing patch {index}: {image_path}")
        for sigma_i in SIGMA_I_VALUES:
            for sigma_x in SIGMA_X_VALUES:
                print(f"Parameters: sigma_i={sigma_i}, sigma_x={sigma_x}")
                output_stem = os.path.join(
                    OUTPUT_DIR,
                    f"{os.path.splitext(filename)[0]}_{sigma_i}_{sigma_x}",
                )
                normalized_cuts_lanczos(filename, image_path, output_stem, k, sigma_i, sigma_x)


if __name__ == "__main__":
    main()
