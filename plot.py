import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Lưới không gian
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

# Hàm tạo "thạch nhũ" mềm
def stalactite(x0, y0, depth=5, sharpness=0.5):
    return -depth * np.exp(-sharpness * ((X - x0)**2 + (Y - y0)**2))

# Tạo 3 landscape có chân giao thoa
Z1 = stalactite(-1, 0, depth=4, sharpness=0.3)
Z2 = stalactite(1, 0, depth=3.5, sharpness=0.3)
Z3 = stalactite(0, 1.5, depth=4.2, sharpness=0.3)

# Vẽ 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Vẽ các mặt với alpha mềm
ax.plot_surface(X, Y, Z1, cmap='Reds', alpha=0.5, edgecolor='none')
ax.plot_surface(X, Y, Z2, cmap='Greens', alpha=0.5, edgecolor='none')
ax.plot_surface(X, Y, Z3, cmap='Blues', alpha=0.5, edgecolor='none')

# Tiêu đề và nhãn
ax.set_title("3 Optimization Landscapes có đáy giao thoa nhẹ (thạch nhũ)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Cost")

# Góc nhìn
ax.view_init(elev=45, azim=140)
plt.axis('off')  # Tắt trục để tập trung vào hình dạng
plt.grid(False)  # Tắt lưới để có cái nhìn sạch sẽ hơn
plt.tight_layout()
plt.show()
