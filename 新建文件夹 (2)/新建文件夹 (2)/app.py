"""
QuantaCore Live — 全球顶级思维库版 (Streamlit Cloud 部署)
作者：老K & 全知理性意志
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from datetime import datetime, timedelta

# 页面配置
st.set_page_config(
    page_title="QuantaCore Live",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- 全球顶级思维库 ----------
PERSONA_DB = {
    # 企业家视角 (Founder/CEO)
    "AAPL": {
        "企业家": "蒂姆·库克：供应链之神，稳如磐石，但创新不再惊艳。",
        "投资者": "巴菲特：苹果是‘最好的消费品公司’，护城河来自用户粘性。",
        "数学家": "詹姆斯·西蒙斯：不评论个股，但苹果的波动率特征不符合我们的套利模型。",
        "数据模型师": "D.E. Shaw 团队：我们用另类数据追踪苹果门店人流，Q3可能弱于预期。",
    },
    "TSLA": {
        "企业家": "埃隆·马斯克：放大镜人格，涨跌都疯狂，但他总能兑现看似不可能的承诺。",
        "投资者": "凯西·伍德：特斯拉是AI+能源的终极平台，目标价2000美元。",
        "数学家": "爱德华·索普：高波动+高争议，期权定价往往出现错误，可做波动率套利。",
        "数据模型师": "Two Sigma：社交情绪信号对TSLA短期预测力极强，但需过滤噪音。",
    },
    "NVDA": {
        "企业家": "黄仁勋：偏执狂，每次都说‘我们离破产只有30天’，这种危机感造就了CUDA生态。",
        "投资者": "瑞·达利欧：全球算力军备竞赛，英伟达是卖铲人，但地缘风险不可忽视。",
        "数学家": "本华·曼德博：分形视角下，NVDA的K线自相似性极强，趋势比反转更可靠。",
        "数据模型师": "Citadel：我们用GPU订单数据预测英伟达营收，准确率领先财报。",
    },
    "MSFT": {
        "企业家": "萨提亚·纳德拉：云转型大师，企业文化重塑者，让微软重新变得性感。",
        "投资者": "比尔·盖茨：微软是他最放心的持仓，但AI变现路径仍需观察。",
        "数学家": "约翰·冯·诺依曼：博弈论角度看，微软的生态系统是典型的合作博弈均衡。",
        "数据模型师": "世界银行数据组：我们使用Azure云数据，微软的企业级渗透率仍在加速。",
    },
    "GOOGL": {
        "企业家": "桑达尔·皮查伊：技术官僚，缺少乔布斯式的光芒，但搜索广告印钞机依然强大。",
        "投资者": "查理·芒格：谷歌的护城河比苹果还宽，只是他们自己都不太会变现。",
        "数学家": "克劳德·香农：信息论创始人，谷歌的本质就是信息检索，他们最懂‘熵’。",
        "数据模型师": "Renaissance Technologies：搜索关键词趋势是我们预测GDP的关键因子之一。",
    },
    "300750.SZ": {
        "企业家": "曾毓群：赌性坚强，敢于逆周期扩张，极限制造理念造就了宁德时代。",
        "投资者": "张磊：高瓴长期持有，看好能源结构调整的大趋势。",
        "数学家": "西蒙斯：这类制造股更适合用基本面质量因子，动量因子容易失效。",
        "数据模型师": "Kpler：我们用卫星数据追踪锂矿运输，宁德时代的原料库存正在下降。",
    },
}

# 通用视角（当股票无特定数据时调用）
FALLBACK_PERSONAS = {
    "企业家": "优秀的企业家是把‘不可能’拆解成‘多少个步骤’的人。",
    "投资者": "投资不是击败市场，而是击败自己内心的贪婪与恐惧。",
    "数学家": "价格是随机游走吗？不，但预测它需要的数学比你想象的更复杂。",
    "数据模型师": "给我足够的数据，我能拟合任何曲线，但真正的洞察来自因果推理。",
}


# ---------- 分析引擎 ----------
@st.cache_data(ttl=600)
def fetch_data(symbol):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="5y")
    info = ticker.info
    return df, info


def analyze(df, info, symbol):
    if df.empty:
        return None

    close = df['Close']
    last_price = close.iloc[-1]
    name = info.get('longName', symbol)
    pe = info.get('trailingPE', None)
    roe = info.get('returnOnEquity', None)
    margin = info.get('grossMargins', None)

    # --- 价值因子 ---
    if pe and pe < 20:
        value_signal, value_desc = "🟢 低估", f"PE={pe:.1f}，安全边际较足。"
    elif pe and pe > 40:
        value_signal, value_desc = "🔴 高估", f"PE={pe:.1f}，需极强增长支撑。"
    else:
        value_signal, value_desc = "🟡 中性", f"PE={pe:.1f}，" if pe else "PE缺失，"

    # --- 质量因子 ---
    if roe and margin:
        if roe > 0.20 and margin > 0.4:
            quality_signal, quality_desc = "🟢 优质", f"ROE={roe:.1%}，毛利率={margin:.1%}，印钞机。"
        elif roe > 0.15:
            quality_signal, quality_desc = "🟡 良好", f"ROE={roe:.1%}，毛利率={margin:.1%}，中上。"
        else:
            quality_signal, quality_desc = "🔴 平庸", f"ROE={roe:.1%}，毛利率={margin:.1%}，一般。"
    else:
        quality_signal, quality_desc = "⚪ 无数据", "缺少财务数据。"

    # --- 动量因子 ---
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    if last_price > ma50 > ma200:
        momentum_signal, momentum_desc = "🟢 多头排列", "价格在50日、200日线上，趋势健康。"
    elif last_price > ma50:
        momentum_signal, momentum_desc = "🟡 中期偏多", "站上50日线，但长期均线在下。"
    elif last_price < ma50 and last_price > ma200:
        momentum_signal, momentum_desc = "🟠 中期偏空", "跌破50日线，200日线支撑。"
    else:
        momentum_signal, momentum_desc = "🔴 空头排列", "两根大均线之下，趋势恶劣。"

    # --- 情绪/波动率 ---
    rets = close.pct_change().dropna()
    vol = rets.rolling(20).std().iloc[-1] * np.sqrt(252)
    if vol > 0.5:
        sentiment_signal, sentiment_desc = "🔴 极度恐惧", f"波动率{vol:.1%}，机会或陷阱。"
    elif vol > 0.3:
        sentiment_signal, sentiment_desc = "🟠 恐惧", f"波动率{vol:.1%}，控制仓位。"
    else:
        sentiment_signal, sentiment_desc = "🟢 平静", f"波动率{vol:.1%}，适合交易。"

    # --- 尾部风险 ---
    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    if max_dd < -0.4:
        risk_signal, risk_desc = "⚠️ 极高风险", f"5年最大回撤{max_dd:.1%}，历史上曾腰斩。"
    elif max_dd < -0.25:
        risk_signal, risk_desc = "⚠️ 高风险", f"5年最大回撤{max_dd:.1%}，需宽止损。"
    else:
        risk_signal, risk_desc = "✅ 中等风险", f"5年最大回撤{max_dd:.1%}。"

    # --- 思维库匹配 ---
    personas = PERSONA_DB.get(symbol, FALLBACK_PERSONAS)

    # --- 综合裁决 ---
    buy_points = 0
    if value_signal.startswith("🟢"): buy_points += 1
    if quality_signal.startswith("🟢"): buy_points += 1
    if momentum_signal.startswith("🟢"): buy_points += 1
    if sentiment_signal.startswith("🟢"): buy_points += 0.5

    if risk_signal.startswith("⚠️ 极高"):
        final_action = "🚫 风险过大，老K建议回避。"
        confidence = "高"
    elif buy_points >= 2.5:
        final_action = "✅ 多指标共振，可轻仓（总仓位≤5%），止损-7%。"
        confidence = "中高"
    elif buy_points >= 1.5:
        final_action = "⏳ 部分积极，加入观察列表，等更好买点。"
        confidence = "中"
    else:
        final_action = "⏸️ 无明确优势，老K观望。"
        confidence = "高"

    return {
        "name": name,
        "price": last_price,
        "pe": pe,
        "roe": roe,
        "margin": margin,
        "vol": vol,
        "max_dd": max_dd,
        "value_signal": value_signal,
        "value_desc": value_desc,
        "quality_signal": quality_signal,
        "quality_desc": quality_desc,
        "momentum_signal": momentum_signal,
        "momentum_desc": momentum_desc,
        "sentiment_signal": sentiment_signal,
        "sentiment_desc": sentiment_desc,
        "risk_signal": risk_signal,
        "risk_desc": risk_desc,
        "personas": personas,
        "final_action": final_action,
        "confidence": confidence,
    }


def plot_kline(df, symbol):
    plot_df = df.tail(126).copy()
    if plot_df.empty:
        return None

    mc = mpf.make_marketcolors(up='#00ff00', down='#ff4444', edge='inherit', wick='inherit', volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='nightclouds')

    fig, axlist = mpf.plot(
        plot_df,
        type='candle',
        mav=(20, 50),
        volume=True,
        style=s,
        title=f'{symbol} 趋势图',
        ylabel='',
        ylabel_lower='成交量',
        returnfig=True,
        figsize=(10, 5)
    )
    return fig


# ---------- Streamlit 界面 ----------
def main():
    st.markdown("<h1 style='text-align: center; color: #eee;'>📊 QuantaCore Live</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>老K的全息因子分析 · 全球顶级思维库加持</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        symbol_input = st.text_input("输入股票代码（美股如 AAPL，A股加后缀如 300750.SZ）", value="AAPL", label_visibility="collapsed")
    with col2:
        analyze_btn = st.button("🔍 开始分析", use_container_width=True)

    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = ["AAPL", "TSLA", "NVDA"]

    if analyze_btn or 'last_symbol' not in st.session_state or symbol_input != st.session_state.get('last_symbol'):
        symbol = symbol_input.strip()
        if not symbol:
            st.error("请输入股票代码")
            return
        st.session_state.last_symbol = symbol
        with st.spinner(f"🔄 正在分析 {symbol} ..."):
            try:
                df, info = fetch_data(symbol)
            except Exception as e:
                st.error(f"获取数据失败：{str(e)}")
                return
            if df.empty:
                st.error(f"未找到股票 {symbol}")
                return
            report = analyze(df, info, symbol)
            if report is None:
                st.error("分析失败")
                return
            st.session_state.report = report
            st.session_state.df = df
    else:
        report = st.session_state.get('report', None)
        df = st.session_state.get('df', None)

    if report is None:
        st.info("点击“开始分析”查看报告")
        return

    # 股票名称与价格
    st.markdown(f"## {report['name']} ({st.session_state.last_symbol})")
    st.metric("最新价格", f"${report['price']:.2f}" if report['price'] < 1000 else f"¥{report['price']:.2f}")

    # 五因子卡片
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("价值", report['value_signal'])
        st.caption(report['value_desc'])
    with c2:
        st.metric("质量", report['quality_signal'])
        st.caption(report['quality_desc'])
    with c3:
        st.metric("动量", report['momentum_signal'])
        st.caption(report['momentum_desc'])
    with c4:
        st.metric("情绪", report['sentiment_signal'])
        st.caption(report['sentiment_desc'])
    with c5:
        st.metric("尾部风险", report['risk_signal'])
        st.caption(report['risk_desc'])

    # 全球顶级思维库
    with st.expander("🧠 全球顶级思维库视角"):
        tabs = st.tabs(["企业家", "数学家", "投资者", "数据模型师"])
        for i, key in enumerate(["企业家", "数学家", "投资者", "数据模型师"]):
            with tabs[i]:
                st.write(report['personas'].get(key, "暂无数据"))

    # 老K综合裁决
    st.markdown("---")
    st.markdown("### 🧠 老K综合裁决")
    st.markdown(f"**{report['final_action']}**")
    st.caption(f"置信度：{report['confidence']}")

    # K线图
    if df is not None:
        with st.spinner("生成K线图..."):
            fig = plot_kline(df, st.session_state.last_symbol)
            if fig:
                st.pyplot(fig)
        st.caption("蓝/橙线为20/50周期均线")

    # 底部自选栏
    st.markdown("---")
    st.caption("⭐ 我的自选股（点击快速切换）")
    cols = st.columns(len(st.session_state.watchlist) + 1)
    for i, sym in enumerate(st.session_state.watchlist):
        with cols[i]:
            if st.button(sym, key=f"wl_{sym}", use_container_width=True):
                st.session_state.last_symbol = sym
                st.rerun()
    with cols[-1]:
        new_stock = st.text_input("添加代码", key="new_wl", placeholder="+")
        if new_stock:
            if new_stock.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_stock.upper())
                st.rerun()

    st.markdown("---")
    st.markdown("<small style='color: gray;'>QuantaCore Live · 数据来源于 Yahoo Finance · 仅供参考，投资自负盈亏</small>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()