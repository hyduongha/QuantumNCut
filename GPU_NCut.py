# Lượng tử cho nhân ma trận với vector ngay tại hàm def A_mul(x) của thuật toán Lanczos

from datetime import datetime
import time
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as splinalg
from sklearn.cluster import KMeans
from skimage import io, color
import os
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def compute_weight_matrix_coo_knn_gpu(image, sigma_i, sigma_x, k_neighbors=10):
    """Tính ma trận trọng số song song trên GPU, không dùng vòng lặp."""
    h, w, c = image.shape
    coords = np.array(np.meshgrid(range(h), range(w))).reshape(2, -1).T
    features = image.reshape(-1, c)

    knn = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree').fit(coords)
    distances, indices = knn.kneighbors(coords)

    # Dữ liệu cho sparse COO
    row_idx = cp.asarray(np.repeat(np.arange(len(coords)), k_neighbors), dtype=cp.int32)
    col_idx = cp.asarray(indices.flatten(), dtype=cp.int32)

    features_gpu = cp.array(features, dtype=cp.float32)
    distances_gpu = cp.array(distances, dtype=cp.float32).reshape(-1)

    # Tạo matrix feature cho tất cả (N x K x C)
    idx_row_expand = cp.repeat(cp.arange(features_gpu.shape[0]), k_neighbors)
    idx_col_expand = col_idx

    # Lấy feature từng điểm và neighbor tương ứng (vector hóa)
    features_row = features_gpu[idx_row_expand]  # (N*K, C)
    features_col = features_gpu[idx_col_expand]  # (N*K, C)

    diff = features_row - features_col
    W_feature = cp.exp(-cp.sum(diff ** 2, axis=1) / (2 * sigma_i ** 2))  # (N*K,)
    W_space = cp.exp(-distances_gpu ** 2 / (2 * sigma_x ** 2))           # (N*K,)

    values = W_feature * W_space

    W_sparse = sp.coo_matrix((values, (row_idx, col_idx)), shape=(len(coords), len(coords)))
    return W_sparse

def compute_laplacian_coo(W_coo):
    D = cp.array(W_coo.sum(axis=1)).flatten()
    D_coo = sp.coo_matrix((D, (cp.arange(len(D)), cp.arange(len(D)))), shape=W_coo.shape)
    L_coo = D_coo - W_coo
    return L_coo, D_coo

def compute_ncut_lanczos(W_coo, k=2, max_iter=100, tol=1e-5):
    """
    Tính k vector riêng nhỏ nhất của ma trận chuẩn hóa NCut:
    A = I - D^{-1/2} W D^{-1/2}
    Bằng phương pháp Lanczos tự viết, không dùng eigsh.
    """
    n = W_coo.shape[0]

    # 1. Chuẩn hóa ma trận W
    D_vals = cp.array(W_coo.sum(axis=1)).flatten()
    D_inv_sqrt = 1.0 / cp.sqrt(D_vals + 1e-8)
    row, col = W_coo.row, W_coo.col
    data = W_coo.data * D_inv_sqrt[row] * D_inv_sqrt[col]
    W_norm = sp.coo_matrix((data, (row, col)), shape=W_coo.shape)

    # 2. Ma trận A = I - W_norm (chuẩn hóa)
    def A_mul(x):
        # return x - W_norm @ x
        from qmatmul import qmatmul_qiskit
        Wx = qmatmul_qiskit(W_norm.toarray().get().copy(), x.get().copy())

        res = x.get().astype(np.float64) - Wx
        return cp.array(res, dtype=cp.float32)

    # 3. Khởi tạo vector đầu tiên
    Q = []
    alphas = []
    betas = []

    q = cp.random.randn(n).astype(cp.float32)
    q /= cp.linalg.norm(q)
    Q.append(q)
    beta = 0.0
    q_prev = cp.zeros_like(q)

    for j in range(max_iter): 
        z = A_mul(Q[-1])
        alpha = cp.dot(Q[-1], z)
        alphas.append(alpha)

        z = z - alpha * Q[-1] - beta * q_prev

        # Reorthogonalization đơn giản
        for q_i in Q:
            z -= cp.dot(q_i, z) * q_i

        beta = cp.linalg.norm(z)
        if beta < tol or len(alphas) >= k + 20:
            break

        betas.append(beta)
        q_prev = Q[-1]
        Q.append(z / beta)

    # 4. Ma trận tridiagonal T
    m = len(alphas)
    T = cp.zeros((m, m), dtype=cp.float32)
    for i in range(m):
        T[i, i] = alphas[i]
        if i > 0:
            T[i, i-1] = T[i-1, i] = betas[i-1]

    # 5. Trị riêng & vector riêng của T
    vals, vecs = cp.linalg.eigh(T)

    # Sắp xếp trị riêng tăng dần
    sorted_idx = cp.argsort(vals)
    vecs = vecs[:, sorted_idx]

    # Loại trị riêng ≈ 0
    nonzero = cp.where(cp.abs(vals[sorted_idx]) > 1e-5)[0]
    vecs = vecs[:, nonzero[:k]]

    # 6. Trả về vector riêng trong không gian gốc
    Q_mat = cp.stack(Q, axis=1)
    eigenvectors = Q_mat @ vecs  # n x k

    return eigenvectors

def assign_labels(eigen_vectors, k):
    return KMeans(n_clusters=k, random_state=0).fit(eigen_vectors.get()).labels_

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

    start = time.perf_counter() 

    W_coo = compute_weight_matrix_coo_knn_gpu(image, sigma_i, sigma_x)
    vecs = compute_ncut_lanczos(W_coo, k)

    end = time.perf_counter()
    print("Thời gian thực thi Ncut:", end - start, "giây")

    labels = assign_labels(vecs, k)
    save_segmentation(image, labels, k, output_path + ".jpg")
    save_seg_file(labels.reshape(image.shape[:2]), image.shape, output_path + ".seg", imagename)
    del W_coo, vecs
    cp.get_default_memory_pool().free_all_blocks()

    return start, end

def main():
    # input_path = "G:\Ket qua Nhu Y_dang lam\\4x8_bigsize\split_image_Hai"
    # output_path = "G:\Ket qua Nhu Y_dang lam\\4x8_bigsize\split_image_segmentation_Hai"
    # excel_path = os.path.join(output_path, "G:\Ket qua Nhu Y_dang lam\\4x8_bigsize\log.xlsx")  # file Excel lưu 3 cột

    input_path = ".\data\split_image_Hai"
    output_path = ".\data\split_image_segmentation_Hai"
    excel_path = os.path.join(output_path, ".\data\log.xlsx")  # file Excel lưu
    if not os.path.isdir(input_path):
        print(f"❌ Thư mục {input_path} không tồn tại!")
        exit()
    
    image_files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    if not image_files:
        print(f"❌ Không tìm thấy file ảnh nào trong {input_path}!")
        exit()

    log_rows = []  # mỗi phần tử: (tên file, bắt đầu, kết thúc)

    for idx, file_name in enumerate(image_files, start=1):
        k = 3
        image_path = os.path.join(input_path, file_name)
        print(f"📷 Đang xử lý ảnh {idx}: {image_path}")

        sigma_i = 0.009
        sigma_x = 8
        
        save_image_name = os.path.join(output_path, f"{os.path.splitext(file_name)[0]}")

        bat_dau, ket_thuc = normalized_cuts_eigsh(file_name, image_path, save_image_name, k, sigma_i, sigma_x)
        log_rows.append((file_name, bat_dau, ket_thuc))
    
    # Ghi một lần sau cùng
    if log_rows:
        new_df = pd.DataFrame(log_rows, columns=["Tên file", "Bắt đầu", "Kết thúc"])
        if os.path.exists(excel_path):
            combined = pd.concat([pd.read_excel(excel_path), new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_excel(excel_path, index=False)
        print(f"📝 Đã ghi log vào: {excel_path}")

if __name__ == "__main__":
    main()
