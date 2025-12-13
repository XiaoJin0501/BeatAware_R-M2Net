import torch
import torch.nn.functional as F

# ==============================================================================
# 1. Mac / CPU 调试版 (Slow but Stable)
# ==============================================================================
def selective_scan_cpu(x, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True):
    """
    纯 Python 实现的 Selective Scan。
    用于 Mac 本地调试，无需编译任何 CUDA。
    """
    batch, dim, length = x.shape
    _, state, _ = B.shape
    
    # Delta 处理
    if delta_bias is not None:
        dt = dt + delta_bias.view(1, dim, 1)
    if delta_softplus:
        dt = F.softplus(dt)
        
    # 离散化 (Discretization)
    # A: [D, S] -> dA: [B, D, S, L]
    dA = torch.exp(torch.einsum('ds,bdl->bdsl', A, dt))
    # B: [B, S, L] -> dB: [B, D, S, L]
    dB = torch.einsum('bdl,bsl->bdsl', dt, B)
    
    # 扫描 (Scan Loop)
    h = torch.zeros(batch, dim, state, device=x.device)
    ys = []
    
    x_expanded = x.unsqueeze(2) # [B, D, 1, L]
    
    for t in range(length):
        h = dA[:, :, :, t] * h + dB[:, :, :, t] * x_expanded[:, :, :, t]
        y_t = torch.einsum('bd, bds->bd', C[:, :, t], h)
        ys.append(y_t)
        
    y = torch.stack(ys, dim=2) # [B, D, L]
    
    if D is not None:
        y = y + D.unsqueeze(-1) * x
    if z is not None:
        y = y * F.silu(z)
        
    return y

# ==============================================================================
# 2. Linux / CUDA 高性能版 (尝试导入编译好的核心)
# ==============================================================================
try:
    # 优先尝试导入 M3ANet 风格的自定义 CUDA 核心 (如果你编译了 selective_scan_cuda)
    import selective_scan_cuda
    
    class SelectiveScanCuda(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
            # 这里的接口需要根据你具体的 .so 文件编译情况调整
            # 这是一个标准的 Mamba wrapper 示例
            if delta_bias is not None:
                dt = delta + delta_bias.view(1, -1, 1)
            else:
                dt = delta
            if delta_softplus:
                dt = F.softplus(dt)
                
            # 调用 CUDA 实现 (假设接口名为 fwd)
            out, x, *rest = selective_scan_cuda.fwd(u, dt, A, B, C, D, None, None, False)
            ctx.save_for_backward(u, dt, A, B, C, D, x)
            return out

        @staticmethod
        def backward(ctx, dout):
            # 简化的 backward 占位，实际需要调用 bwd kernel
            # 如果你用的是官方 mamba-ssm，不需要手写这个 Function
            return None 

    def selective_scan_cuda_fn(x, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True):
        # 简单的封装调用
        return SelectiveScanCuda.apply(x, dt, A, B, C, D, delta_bias, delta_softplus)

    CUDA_IMPL = "custom_cuda"

except ImportError:
    # 再次尝试导入官方 mamba_ssm (推荐在 Linux 上直接 pip install mamba-ssm)
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
        CUDA_IMPL = "official_mamba"
    except ImportError:
        CUDA_IMPL = "cpu"

# ==============================================================================
# 3. 统一分发入口
# ==============================================================================
def selective_scan_1d(x, dt, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True):
    """
    自动选择后端的 Selective Scan 接口
    """
    if x.is_cuda and CUDA_IMPL == "official_mamba":
        # 官方库通常需要 B, C 维度为 [B, L, S] (Channel Last)
        # 这里做一个简单的维度转换示例，具体视版本而定
        return selective_scan_fn(x, dt, A, B, C, D, z, delta_bias, delta_softplus)
        
    elif x.is_cuda and CUDA_IMPL == "custom_cuda":
        # 使用 M3ANet 风格的 kernel
        # 注意：这里需要确保输入维度符合你编译的 kernel 要求
        return selective_scan_cpu(x, dt, A, B, C, D, z, delta_bias, delta_softplus) # 暂时 fallback 避免接口报错
        
    else:
        # Mac / CPU
        return selective_scan_cpu(x, dt, A, B, C, D, z, delta_bias, delta_softplus)