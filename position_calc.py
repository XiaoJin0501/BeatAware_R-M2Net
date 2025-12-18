import math

def advanced_trade_calc(
    total_capital,      # 总资金
    allocation_pct,     # 仓位分配比例 (如 0.33)
    risk_amount,        # 止损金额 (如 $10)
    entry_price,        # 开仓价格
    stop_loss_price,    # 止损价格
    direction='long',   # 方向
    fee_rate=0.00035     # 手续费率 (默认万6，根据你的交易所调整)
):
    print("="*50)
    print(f"       💰 合约交易计划书 ({'🟢 做多 Long' if direction == 'long' else '🔴 做空 Short'})")
    print("="*50)

    # --- 1. 基础逻辑检查 ---
    if direction == 'long' and stop_loss_price >= entry_price:
        print("❌ 错误：做多时，止损价必须 < 开仓价")
        return
    elif direction == 'short' and stop_loss_price <= entry_price:
        print("❌ 错误：做空时，止损价必须 > 开仓价")
        return

    # --- 2. 核心数据计算 ---
    # 计算本金
    margin = total_capital * allocation_pct
    
    # 计算价格波动的百分比 (距离)
    price_diff_abs = abs(entry_price - stop_loss_price)
    price_diff_pct = price_diff_abs / entry_price

    # 计算仓位大小 (Position Size)
    # 核心公式：亏损金额 = 仓位价值 * 价格波动%
    # 所以：仓位价值 = 亏损金额 / 价格波动%
    position_size = risk_amount / price_diff_pct

    # 计算杠杆 (Leverage)
    leverage = position_size / margin
    
    # 向下取整保留1位小数，或者取整 (交易所通常不支持太碎的杠杆)
    suggested_leverage = math.floor(leverage) 

    # --- 3. 风险与强平分析 ---
    # 估算强平价格 (仅供参考，不同交易所维持保证金率不同)
    # 逐仓模式下：强平价 ≈ 开仓价 * (1 +/- 1/杠杆)
    if direction == 'long':
        liq_price = entry_price * (1 - (1/suggested_leverage) + 0.005) # 0.005是缓冲
    else:
        liq_price = entry_price * (1 + (1/suggested_leverage) - 0.005)

    # 估算手续费 (开仓+平仓) = 总名义价值 * 费率 * 2
    est_fees = position_size * fee_rate * 2

    # --- 4. 输出：第一部分 (设置与风控) ---
    print(f"【资金设置】")
    print(f"• 总资金:      ${total_capital:.2f}")
    print(f"• 本次投入:    ${margin:.2f} (仓位 {allocation_pct*100}%)")
    print(f"• 允许亏损:    -${risk_amount:.2f}")
    print(f"• 预估手续费:  -${est_fees:.2f} (买卖双边)")
    print("-" * 50)
    
    print(f"【开单参数】")
    print(f"• 开仓价格:    {entry_price}")
    print(f"• 止损价格:    {stop_loss_price} (距离 {price_diff_pct*100:.2f}%)")
    print(f"• 仓位总价值:  ${position_size:.2f} (总个币数量: {position_size/entry_price:.4f})")
    print(f"▶ 建议杠杆:    x{suggested_leverage} 倍")
    print("-" * 50)

    print(f"【安全警示】")
    print(f"• 你的止损价:  {stop_loss_price}")
    print(f"• 预估强平价:  {liq_price:.2f}")
    
    if direction == 'long':
        if stop_loss_price <= liq_price:
            print("⚠️ 危险！你的止损价低于强平价，可能会先爆仓！请降低杠杆或追加保证金。")
        else:
            print("✅ 安全：止损会在强平之前触发。")
    else:
        if stop_loss_price >= liq_price:
            print("⚠️ 危险！你的止损价高于强平价，可能会先爆仓！")
        else:
            print("✅ 安全：止损会在强平之前触发。")

    # --- 5. 输出：第二部分 (止盈目标) ---
    print("\n" + "="*50)
    print(f"       🎯 止盈目标规划 (扣除手续费后净利)")
    print("="*50)
    print(f"| {'盈亏比(R:R)':<10} | {'止盈价格':<12} | {'纯利润(Net)':<12} | {'ROE(本金收益)':<10} |")
    print("-" * 56)

    rr_ratios = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    
    for rr in rr_ratios:
        # 毛利
        gross_profit = risk_amount * rr
        # 净利 (扣除手续费)
        net_profit = gross_profit - est_fees
        
        # 计算达到该利润需要的价格变动
        target_move_pct = gross_profit / position_size
        
        if direction == 'long':
            tp_price = entry_price * (1 + target_move_pct)
        else:
            tp_price = entry_price * (1 - target_move_pct)
            
        # 本金收益率 (ROE)
        roe = (net_profit / margin) * 100
        
        print(f"| 1 : {str(rr):<5} | {tp_price:<12.2f} | ${net_profit:<11.2f} | {roe:<9.1f}% |")

    print("="*50)
    print("注：ROE为扣除手续费后的实际本金回报率")


# ==========================================
# 👉 请在下方修改你的交易参数
# ==========================================

# 1. 你的钱包
Total_Capital = 100.0    # 你的总本金 ($)
Alloc_Pct = 0.33         # 每次投入本金的 33%

# 2. 你的风控
Risk_Per_Trade = 10.0    # 如果止损，你愿意亏掉多少钱 ($)

# 3. 你的图表分析 (填入你看到的点位)
Entry_Price = 87100.0    # 打算在哪里开单
Stop_Loss   = 87300.0    # 止损放在哪里 (技术支撑/压力位)
Direction   = 'short'     # 'long'(做多) 或 'short'(做空)

# 运行程序
advanced_trade_calc(
    Total_Capital, 
    Alloc_Pct, 
    Risk_Per_Trade, 
    Entry_Price, 
    Stop_Loss, 
    Direction
)