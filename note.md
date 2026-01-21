L là ma trận đầu vào = D - W
=> L có n trị riêng, n trị riêng đều thực, ko âm; L là ma trận đối xứng hermit

1. QPE để estimate trị riêng
=> nếu mình sử dụng pp block-encoding
=> U(L) = block-encoding(L) / Unita
=> trị riêng của U(L) chứa trị riêng của L

QPE chỉ tính được trị riêng phức có độ dài = 1
(quantum phase estimation)
=> trị riêng = a + bi, QPE trả về b, với điều kiện a^2 + b^2 = 1
=> nếu trị riêng của mình là thực => b = 0 => qpe luôn trả về 0
=> Áp dụng qpe không được.

2. Hướng Lanzcos
Lanzcos algorithm (chỉ tính đc trên ma trận hermit)
- Input: Ma trận L: N x N (N cực lớn)
- output: Ma trận T: k x k (k nhỏ) (k là số trị riêng nhỏ nhất cần tìm)
O(N^3) => O(k*N^2)

Các bước trong Lanzcos algorithm:
- L x v = g
L = [[1, 2], v = [1, 2]
        [3, 4]]
=> g[0] = L[0] x v = 1*1 + 2*2 = 5
=> g[1] = L[1] x v = 3*1 + 4*2 = 11

for i in range(L.shape[0]):
    lấy hàng i của L và nhân với v (tích nội) = g[i]
- np.dot(g, v)
...
    classical-dot(L[i], v): O(N) (N là kích thước của L[i[)
    quantum-dot(L[i], v): O(log(N))

=> quantum lanzcos algorithm: O(k*N^2) => O(k*N*log(N))
=> swap-test(u, v) để tính tích nội u, v

Với số chiều của u = N, tốn log2(N) qubits 

3. Thuật toán swap-test
- Có bước prepare_state (tải u/v vào mạch lượng tử) là thuật toán của mình phát triển
