import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def solve_helmholtz_fem(eps_func, d, eps1, eps2, lambda0=1.0, n_trans=1000, n_ext=200):
    k0 = 2 * np.pi / lambda0
    k1 = k0 * np.sqrt(eps1 + 0j)
    k2 = k0 * np.sqrt(eps2 + 0j)
    
    # 1. 建立非均匀网格 (L_ext = 2*lambda0)
    L_ext = 2.0 * lambda0
    z_ext_left = np.linspace(-d/2 - L_ext, -d/2, n_ext, endpoint=False)
    z_trans = np.linspace(-d/2, d/2, n_trans)
    z_ext_right = np.linspace(d/2, d/2 + L_ext, n_ext + 1)[1:]
    
    nodes = np.concatenate([z_ext_left, z_trans, z_ext_right])
    num_nodes = len(nodes)
    num_elements = num_nodes - 1
    
    # 2. 组装全局矩阵 K - k0^2 * M
    # 这里使用 lil_matrix 方便组装，最后转为 csr
    A = lil_matrix((num_nodes, num_nodes), dtype=complex)
    b = np.zeros(num_nodes, dtype=complex)
    
    for i in range(num_elements):
        z_left, z_right = nodes[i], nodes[i+1]
        h = z_right - z_left
        
        # 单元中点的 epsilon (用于积分近似)
        eps_mid = eps_func((z_left + z_right) / 2.0)
        
        # 单元刚度矩阵 (K_el) 和 质量矩阵 (M_el)
        # K_el = 1/h * [[1, -1], [-1, 1]]
        # M_el = h/6 * [[2, 1], [1, 2]] * eps_mid
        ke = (1.0/h) * np.array([[1, -1], [-1, 1]])
        me = (h/6.0) * eps_mid * np.array([[2, 1], [1, 2]])
        
        element_mat = ke - (k0**2) * me
        
        # 填入全局矩阵
        A[i:i+2, i:i+2] += element_mat

    # 3. 施加边界条件 (Robin BC / Port Condition)
    # 左边界 z_min: A[0,0] += j*k1, b[0] = 2j*k1 * E_inc(z_min)
    # 注意入射波相位: E_inc = exp(-j*k1*(z - (-d/2)))，则在 z_min 处相位为 exp(j*k1*L_ext)
    E_inc_at_zmin = np.exp(1j * k1 * L_ext)
    A[0, 0] += 1j * k1
    b[0] += 2j * k1 * E_inc_at_zmin
    
    # 右边界 z_max: A[N,N] += j*k2
    A[-1, -1] += 1j * k2
    
    # 4. 求解线性方程组
    E = spsolve(A.tocsr(), b)
    
    # 5. 提取系数 (相位参考点统一设在过渡层边界)
    # 反射系数 r (在 z = -d/2 处): E(-d/2) = 1 + r
    # 我们找到 nodes == -d/2 的索引
    idx_z1 = n_ext
    r = E[idx_z1] - 1.0
    
    # 透射系数 t (在 z = d/2 处): E(d/2) = t
    idx_z2 = n_ext + n_trans - 1
    t = E[idx_z2]
    
    return nodes, E, r, t, (nodes[idx_z1], nodes[idx_z2])

# ================= 演示：势垒隧穿 (ε从正变负再变正) =================
if __name__ == "__main__":
    def my_eps(z):
        d = 1.0
        # 构造一个“势垒”：中间部分介电常数为负
        if z < -0.2: return 2.25
        elif z > 0.2: return 2.25
        else: return -1.5 # 禁戒区

    # 运行 FEM
    z_nodes, E_field, r, t, (z1, z2) = solve_helmholtz_fem(
        eps_func=np.vectorize(my_eps), d=1.0, eps1=2.25, eps2=2.25, lambda0=1.0)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(z_nodes, np.abs(E_field), 'k', label='|E(z)| (Amplitude)')
    plt.plot(z_nodes, E_field.real, 'b--', alpha=0.5, label='Re{E(z)}')
    
    plt.axvspan(z1, z2, color='red', alpha=0.1, label='Tunneling Barrier')
    plt.title(f'FEM Solution: Reflection r={np.abs(r)**2:.3f}, Transmission t={np.abs(t)**2:.3f}')
    plt.xlabel('z')
    plt.ylabel('Field')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()