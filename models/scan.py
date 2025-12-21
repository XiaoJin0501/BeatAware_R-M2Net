import torch
import torch.nn.functional as F

# ==============================================================================
# 1. 纯 PyTorch 实现 (Slow but Universal)
#    - 适用于: Mac, Windows (无编译环境), Linux (环境配置失败时)
# ==============================================================================
def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """
    Pure PyTorch implementation of Selective Scan.
    Reference: Mamba-SSM (State Space Model)
    Shapes:
        u: (B, D, L)
        delta: (B, D, L)
        A: (D, N)
        B: (B, N, L)
        C: (B, N, L)
        D: (D)
    """
    b, d, l = u.shape
    n = A.shape[1]
    
    # 1. Delta 处理
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)
    
    # 2. 离散化 (Discretization)
    # dt_A = exp(delta * A)  -> (B, D, L, N)
    # dt_B = delta * B       -> (B, D, L, N)
    # A broadcast over L, delta broadcast over N
    delta_a = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    delta_b = torch.einsum('bdl,bnl->bdln', delta, B)
    
    # 3. 扫描 (Scan Loop) - 这是最慢的部分，但在无 CUDA 环境下是必须的
    x = torch.zeros((b, d, n), device=u.device, dtype=u.dtype)
    ys = []
    
    # 扩展 u 以匹配状态维度
    u_unsq = u.unsqueeze(-1) # (B, D, L, 1)
    
    for i in range(l):
        # x[t] = A[t] * x[t-1] + B[t] * u[t]
        x = delta_a[:, :, i] * x + delta_b[:, :, i] * u_unsq[:, :, i]
        
        # y[t] = C[t] * x[t]
        # logic: sum(x * C) over state dimension N
        # x: (B, D, N), C[:,:,i]: (B, N)
        y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
        ys.append(y)
        
    y = torch.stack(ys, dim=2) # (B, D, L)
    
    # 4. 后处理 (Gate & Residual)
    if z is not None:
        y = y * F.silu(z)

    if D is not None:
        y = y + D[..., None] * u
        
    return y

# ==============================================================================
# 2. 自动后端选择器 (Auto-Backend Switch)
# ==============================================================================

# 尝试导入编译好的 CUDA 核心
try:
    import selective_scan_cuda
    USE_CUDA = True
    print("[Scan] ✅ Detected compiled CUDA kernels. Using fast GPU backend.")
except ImportError:
    USE_CUDA = False
    print("[Scan] ⚠️  CUDA kernels not found. Switched to Pure PyTorch backend (CPU/Mac/Win Compatible).")

class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        # 只有当环境完美时才走 CUDA
        if USE_CUDA:
            try:
                # 注意：这里的接口参数顺序可能需要根据实际 selective_scan_cuda 的实现进行微调
                # 这里假设它接受的参数与 Python 版一致
                out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
                ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x)
                ctx.delta_softplus = delta_softplus
                return out
            except Exception as e:
                # 即使导入成功但运行时报错，也回退到 PyTorch
                print(f"[Scan] Runtime CUDA error: {e}. Fallback to PyTorch ref.")
                return selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
        else:
            return selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if USE_CUDA:
            try:
                u, delta, A, B, C, D, z, delta_bias, x = ctx.saved_tensors
                du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                    u, delta, A, B, C, D, z, delta_bias, dout, x, None, None, ctx.delta_softplus, False
                )
                return (du, ddelta, dA, dB, dC, dD, None, ddelta_bias, None, None)
            except Exception:
                # 理论上 Autograd 会自动处理 ref 的 backward，不应进入这里
                return None
        else:
            raise NotImplementedError("Pure PyTorch backward is handled by autograd, should not call this.")

# ==============================================================================
# 3. 统一对外接口
# ==============================================================================
def selective_scan_1d(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """
    对外调用的主函数。
    名字改为 selective_scan_1d 以匹配 ssm.py 中的 import。
    """
    if USE_CUDA:
        return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)
    else:
        # 在纯 PyTorch 模式下，直接调用函数而不是 Function.apply，这样可以保留 PyTorch 的自动求导图
        return selective_scan_ref(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)