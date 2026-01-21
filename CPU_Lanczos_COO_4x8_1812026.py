import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from datetime import datetime
import time
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from skimage import io, color
import os
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import re

def compute_weight_matrix_coo_knn(image, sigma_i, sigma_x, k_neighbors=10):
    h, w, c = image.shape
    N = h * w

    coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T
    features = image.reshape(-1, c)

    knn = NearestNeighbors(n_neighbors=k_neighbors, algorithm="ball_tree")
    knn.fit(coords)
    distances, indices = knn.kneighbors(coords)

    row_idx = np.repeat(np.arange(N), k_neighbors)
    col_idx = indices.flatten()

    feat_row = features[row_idx]
    feat_col = features[col_idx]

    diff = feat_row - feat_col
    W_feature = np.exp(-np.sum(diff ** 2, axis=1) / (2 * sigma_i ** 2))
    W_space = np.exp(-(distances.flatten() ** 2) / (2 * sigma_x ** 2))

    values = W_feature * W_space

    W_sparse = sp.coo_matrix((values, (row_idx, col_idx)), shape=(N, N))
    return W_sparse


def compute_laplacian_coo(W_coo):
    D = np.array(W_coo.sum(axis=1)).flatten()
    D_coo = sp.diags(D)
    L_coo = D_coo - W_coo
    return L_coo, D_coo

def compute_ncut_lanczos(W_coo, k=2, max_iter=30, tol=1e-5):
    n = W_coo.shape[0]

    # --- Normalize W ---
    D_vals = np.array(W_coo.sum(axis=1)).flatten()
    D_inv_sqrt = 1.0 / np.sqrt(D_vals + 1e-8)

    row, col = W_coo.row, W_coo.col
    data = W_coo.data * D_inv_sqrt[row] * D_inv_sqrt[col]
    W_norm = sp.coo_matrix((data, (row, col)), shape=W_coo.shape)

    # --- A = I - W_norm ---
    def A_mul(x):
        # Classical version
        # return x - W_norm @ x

        # Nếu muốn giữ quantum hook:
        from qmatmul import qmatmul_qiskit
        Wx = qmatmul_qiskit(W_norm.toarray(), x)
        return x - Wx

    # --- Lanczos ---
    Q = []
    alphas = []
    betas = []

    q = np.random.randn(n).astype(np.float32)
    q /= np.linalg.norm(q)
    Q.append(q)

    beta = 0.0
    q_prev = np.zeros_like(q)

    for _ in range(max_iter):
        z = A_mul(Q[-1])
        alpha = np.dot(Q[-1], z)
        alphas.append(alpha)

        z = z - alpha * Q[-1] - beta * q_prev

        for qi in Q:
            z -= np.dot(qi, z) * qi

        beta = np.linalg.norm(z)
        if beta < tol or len(alphas) >= k + 20:
            break

        betas.append(beta)
        q_prev = Q[-1]
        Q.append(z / beta)

    # --- Tridiagonal ---
    m = len(alphas)
    T = np.zeros((m, m), dtype=np.float32)

    for i in range(m):
        T[i, i] = alphas[i]
        if i > 0:
            T[i, i - 1] = T[i - 1, i] = betas[i - 1]

    vals, vecs = np.linalg.eigh(T)

    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    nonzero = np.where(np.abs(vals) > 1e-5)[0]
    vecs = vecs[:, nonzero[:k]]

    Q_mat = np.stack(Q, axis=1)
    eigenvectors = Q_mat @ vecs

    return eigenvectors


def assign_labels(eigen_vectors, k):
    return KMeans(n_clusters=k, random_state=0).fit(eigen_vectors).labels_

def save_segmentation(image, labels, k, output_path):
    h, w, c = image.shape
    segmented_image = np.zeros_like(image, dtype=np.uint8)
    for i in range(k):
        mask = labels.reshape(h, w) == i
        cluster_pixels = image[mask]
        mean_color = (cluster_pixels.mean(axis=0) * 255).astype(np.uint8) if len(cluster_pixels) > 0 else np.array([0, 0, 0], dtype=np.uint8)
        segmented_image[mask] = mean_color
    io.imsave(output_path, segmented_image)

def save_seg_file(labels, image_shape, output_path, image_name="image"):
    h, w = image_shape[:2]
    unique_labels = np.unique(labels)
    segments = len(unique_labels)
    
    # Tạo phần header
    header = [
        "format ascii cr",
        f"date {datetime.now().strftime('%a %b %d %H:%M:%S %Y')}",
        f"image ",
        "user 1102",  # Giữ nguyên như file mẫu
        f"width {w}",
        f"height {h}",
        f"segments {segments}",
        "gray 0",
        "invert 0",
        "flipflop 0",
        "data"
    ]
    
    # Tạo dữ liệu pixel theo định dạng (nhãn, dòng, cột bắt đầu, cột kết thúc)
    data_lines = []
    for row in range(h):
        row_labels = labels[row, :]
        start_col = 0
        current_label = row_labels[0]
        
        for col in range(1, w):
            if row_labels[col] != current_label:
                data_lines.append(f"{current_label} {row} {start_col} {col}")
                start_col = col
                current_label = row_labels[col]
        
        # Thêm dòng cuối cùng của hàng
        data_lines.append(f"{current_label} {row} {start_col} {w}")
    
    # Lưu vào file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(data_lines) + "\n")
    
    print(f"✅ File SEG đã lưu: {output_path}")


def normalized_cuts_eigsh(imagename, image_path, output_path, k, sigma_i, sigma_x):
    image = io.imread(image_path)
    image = color.gray2rgb(image) if image.ndim == 2 else image[:, :, :3] if image.shape[2] == 4 else image
    image = image / 255.0

    W_coo = compute_weight_matrix_coo_knn(image, sigma_i, sigma_x)
    start = time.perf_counter()
    vecs = compute_ncut_lanczos(W_coo, k, max_iter=k+20)

    end = time.perf_counter()

    labels = assign_labels(vecs, k)
    save_segmentation(image, labels, k, output_path + ".jpg")
    save_seg_file(labels.reshape(image.shape[:2]), image.shape, output_path + ".seg", imagename)
    del W_coo, vecs

    return start, end

def main():

    excel_path = os.path.join("/content/drive/MyDrive/Test/log1.xlsx")  # file Excel lưu
    input_path = "/content/drive/MyDrive/Test/in1"
    output_path = "/content/drive/MyDrive/Test/out1"
    
    if not os.path.isdir(input_path):
        print(f"❌ Thư mục {input_path} không tồn tại!")
        exit()
    
    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {input_path}!")
        exit()

    log_rows = []  # mỗi phần tử: (tên file, bắt đầu, kết thúc)

    for idx, file_name in enumerate(image_files, start=1):
        k = int(re.search(r"_(\d+)\.png$", file_name).group(1))

        image_path = os.path.join(input_path, file_name)
        print(f"📷 Đang xử lý ảnh {idx}: {image_path}")

        sigma_i = 0.009
        sigma_x = 8
        
        save_image_name = os.path.join(output_path, f"{os.path.splitext(file_name)[0]}")

        bat_dau, ket_thuc = normalized_cuts_eigsh(file_name, image_path, save_image_name, k, sigma_i, sigma_x)
        new_df = pd.DataFrame( [(file_name, bat_dau, ket_thuc)], columns=["Tên file", "Bắt đầu", "Kết thúc"] )

        if os.path.exists(excel_path):
            with pd.ExcelWriter(
                excel_path,
                engine="openpyxl",
                mode="a",
                if_sheet_exists="overlay"
            ) as writer:
                startrow = writer.sheets[next(iter(writer.sheets))].max_row
                new_df.to_excel(
                    writer,
                    index=False,
                    header=False,
                    startrow=startrow
                )
        else:
            new_df.to_excel(excel_path, index=False)

        print(f"📝 Đã ghi tiếp vào: {excel_path}")
    
if __name__ == "__main__":
    main()
