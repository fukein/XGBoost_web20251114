import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import matplotlib  # 导入matplotlib核心库用于字体配置

# ---------------------- 1. 基础配置（重点解决中文显示） ----------------------
# 全局字体设置，优先使用支持中文的字体，适配不同环境
matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS", "Times New Roman"]
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ---------------------- 2. 自定义CSS：确保Streamlit文本中文显示 ----------------------
st.markdown("""
<style>
/* 全局字体设置，确保所有元素支持中文 */
* {
    font-family: "SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Helvetica Neue", Arial, sans-serif !important;
}

body {
    background-color: #f5f7fa;
}

.card {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    padding: 20px;
    margin-bottom: 20px;
}

.section-title {
    font-size: 18px;
    font-weight: bold;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
    margin-bottom: 15px;
}

.label-col {
    text-align: left !important;
    width: 220px;
    padding-right: 10px;
    font-size: 13px;
}

.input-col {
    flex: 1;
}

.result-card {
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    color: white;
    font-weight: bold;
}

.confidence-bar {
    height: 20px;
    border-radius: 10px;
    margin: 5px 0;
    background-color: #e0e0e0;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
}

/* 蓝色预测按钮 */
.stButton>button {
    background-color: #3498db !important;
    color: white !important;
    text-align: center !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
    border: 2px solid white !important;
}
.stButton>button:hover {
    background-color: #2980b9 !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------- 3. 加载模型 & 定义特征范围 ----------------------
# 加载XGBoost模型
try:
    model = joblib.load('XGBoost.pkl')
    st.success("XGBoost模型加载成功!")
except FileNotFoundError:
    st.error("模型文件未找到! 请确保'XGBoost.pkl'在当前目录下。")
    st.stop()

# 定义四个输入特征的范围
feature_ranges = {
    '拉力': {"type": "numerical", "min": 0.0, "max": 999999999.0, "default": 18.8, "step": 0.1},
    '株高': {"type": "numerical", "min": 0.0, "max": 999999999.0, "default": 108.0, "step": 1.0},
    '叶柄长': {"type": "numerical", "min": 0.0, "max": 999999999.0, "default": 17.0, "step": 0.1},
    '节数': {"type": "numerical", "min": 0.0, "max": 999999999.0, "default": 17.0, "step": 1.0}
}

# 模型需要的6个特征名称（顺序必须与模型训练时一致）
model_feature_names = ['拉力', '株高', '叶柄长', '节数', '株高拉力比', '叶柄节数比']

# 倒伏级别说明（可根据实际情况调整）
lodging_levels = {
    0: {"name": "无倒伏", "description": "作物直立生长，无明显倾斜现象，抗倒伏能力强"},
    1: {"name": "轻度倒伏", "description": "作物倾斜角度小于30°，对产量影响较小，抗倒伏能力较强"},
    2: {"name": "中度倒伏", "description": "作物倾斜角度30°-60°，对产量有一定影响，抗倒伏能力中等"},
    3: {"name": "重度倒伏", "description": "作物倾斜角度大于60°，严重影响产量，抗倒伏能力弱"}
}


# ---------------------- 4. 页面结构 ----------------------
st.title("作物倒伏级别预测系统", anchor=False)
st.markdown("请输入作物的相关特征参数，系统将预测其倒伏级别并展示特征贡献度。")

# 特征输入模块（2列布局）
with st.container():
    st.markdown('<div class="card"><h3 class="section-title">作物特征参数</h3>', unsafe_allow_html=True)
    cols = st.columns(2)
    feature_values = {}
    feature_names = list(feature_ranges.keys())

    for idx, (feature, props) in enumerate(feature_ranges.items()):
        with cols[idx % 2]:
            st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 15px;"><div class="label-col">{feature}</div><div class="input-col">', unsafe_allow_html=True)
            if props["type"] == "numerical":
                value = st.number_input(
                    feature,
                    min_value=float(props["min"]),
                    max_value=float(props["max"]),
                    value=float(props["default"]),
                    step=props["step"],
                    format="%.1f" if props["step"] < 1 else "%.0f",
                    label_visibility="collapsed"
                )
            feature_values[feature] = value
            st.markdown('</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------- 5. 预测与SHAP可视化 ----------------------
if st.button("预测倒伏级别", type="primary", use_container_width=True, key="predict_btn"):
    # 计算衍生特征
    try:
        if feature_values['拉力'] == 0:
            st.error("拉力不能为0，请输入有效的拉力值！")
            st.stop()
        if feature_values['节数'] == 0:
            st.error("节数不能为0，请输入有效的节数值！")
            st.stop()
            
        株高拉力比 = feature_values['株高'] / feature_values['拉力']
        叶柄节数比 = feature_values['叶柄长'] / feature_values['节数']
    except Exception as e:
        st.error(f"特征计算错误: {str(e)}")
        st.stop()
    
    # 准备模型输入数据
    model_input = [
        feature_values['拉力'],
        feature_values['株高'],
        feature_values['叶柄长'],
        feature_values['节数'],
        株高拉力比,
        叶柄节数比
    ]
    input_data = pd.DataFrame([model_input], columns=model_feature_names)
    
    # 模型预测
    try:
        pred_class = model.predict(input_data)[0]
        pred_proba = model.predict_proba(input_data)[0]
        # 计算预测置信度（预测类别的概率）
        confidence = pred_proba[int(pred_class)] * 100
    except Exception as e:
        st.error(f"模型预测错误: {str(e)}")
        st.stop()

    # 计算SHAP值（强制字体配置）
    try:
        # 为SHAP单独设置字体，防止被内部配置覆盖
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        base_value = explainer.expected_value
        
        # 处理多分类SHAP值
        if isinstance(shap_values, list):
            single_shap = shap_values[int(pred_class)][0]  # 取预测类别的第一个样本
            if isinstance(base_value, list):
                base_value = base_value[int(pred_class)]
        else:
            single_shap = shap_values[0]  # 二分类情况
            
    except Exception as e:
        st.warning(f"SHAP值计算失败: {str(e)}，将不显示特征贡献图")
        single_shap = None
        base_value = None

    # 存储结果
    st.session_state.pred_results = {
        "pred_class": pred_class,
        "pred_proba": pred_proba,
        "confidence": confidence,  # 预测置信度
        "single_shap": single_shap,
        "feature_names": model_feature_names,
        "feature_values": model_input,
        "base_value": base_value,
        "input_features": feature_values,
        "derived_features": {
            "株高拉力比": 株高拉力比,
            "叶柄节数比": 叶柄节数比
        }
    }

# 显示预测结果
if "pred_results" in st.session_state:
    res = st.session_state.pred_results
    pred_class = int(res["pred_class"])
    
    # 显示输入特征和计算的衍生特征
    st.markdown("### 输入特征与计算特征")
    with st.expander("查看详细特征值", expanded=True):
        input_df = pd.DataFrame(list(res["input_features"].items()), columns=["特征", "值"])
        st.dataframe(input_df, use_container_width=True)
        
        derived_df = pd.DataFrame(list(res["derived_features"].items()), columns=["特征", "值"])
        derived_df["值"] = derived_df["值"].apply(lambda x: f"{x:.4f}")
        st.dataframe(derived_df, use_container_width=True)

    # 显示预测结果（带置信度）
    st.markdown("### 倒伏级别预测结果")
    
    # 获取级别信息，默认为未知级别
    level_info = lodging_levels.get(pred_class, {"name": f"级别{pred_class}", "description": "无详细说明"})
    
    # 定义级别颜色
    level_colors = {
        0: "#2ecc71",  # 绿色-无倒伏
        1: "#3498db",  # 蓝色-轻度
        2: "#f39c12",  # 橙色-中度
        3: "#e74c3c"   # 红色-重度
    }
    level_color = level_colors.get(pred_class, "#95a5a6")  # 默认灰色
    
    # 显示主预测结果
    st.markdown(f"""
    <div class="result-card" style="background-color: {level_color};">
        预测倒伏级别: {pred_class} ({level_info['name']})
    </div>
    <div style="margin: 10px 0;">
        <p><strong>级别描述:</strong> {level_info['description']}</p>
        <p><strong>预测置信度:</strong> {res['confidence']:.2f}%</p>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {res['confidence']}%; background-color: {level_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示各级别概率分布
    st.markdown("#### 各级别概率分布")
    prob_data = []
    for i, prob in enumerate(res["pred_proba"]):
        level_name = lodging_levels.get(i, {"name": f"级别{i}"})["name"]
        prob_data.append({
            "倒伏级别": f"{i} ({level_name})",
            "概率": f"{prob*100:.2f}%",
            "概率值": prob
        })
    
    # 转换为DataFrame并排序
    prob_df = pd.DataFrame(prob_data)
    prob_df = prob_df.sort_values("概率值", ascending=False).drop("概率值", axis=1)
    st.dataframe(prob_df, use_container_width=True)

    # 显示SHAP瀑布图（强制字体配置）
    if res["single_shap"] is not None and res["base_value"] is not None:
        st.markdown("### SHAP特征贡献图")
        st.markdown("蓝色 = 降低倒伏风险，红色 = 增加倒伏风险，长度 = 贡献程度（前10个特征）")
        
        # 再次强制设置字体，确保绘图时生效
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
        
        shap_exp = shap.Explanation(
            values=res['single_shap'],
            base_values=res['base_value'],
            data=res['feature_values'],
            feature_names=res['feature_names']
        )

        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap_exp, max_display=10, show=False)
        
        # 单独设置坐标轴和标签字体（关键：防止SHAP内部覆盖）
        for ax in plt.gcf().axes:
            # 设置标题字体
            ax.set_title(ax.get_title(), fontproperties="SimHei", fontsize=12)
            # 设置坐标轴标签字体
            ax.set_xlabel(ax.get_xlabel(), fontproperties="SimHei", fontsize=10)
            ax.set_ylabel(ax.get_ylabel(), fontproperties="SimHei", fontsize=10)
            # 设置刻度标签字体
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties("SimHei")
                label.set_fontsize(9)
        
        plt.tight_layout()
        st.pyplot(plt.gcf())

        # 显示所有特征的SHAP贡献值
        if st.checkbox("显示所有特征的SHAP值", key="show_shap"):
            shap_df = pd.DataFrame({
                "特征": res['feature_names'],
                "数值": res['feature_values'],
                "SHAP值（贡献度）": res['single_shap'].round(4)
            })
            shap_df["绝对贡献度"] = shap_df["SHAP值（贡献度）"].abs()
            shap_df_sorted = shap_df.sort_values("绝对贡献度", ascending=False).drop("绝对贡献度", axis=1)
            st.dataframe(shap_df_sorted, use_container_width=True)



