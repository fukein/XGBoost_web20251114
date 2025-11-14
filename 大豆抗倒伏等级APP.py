import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import matplotlib
from hashlib import sha256  # 用于密码加密存储

# ---------------------- 1. 基础配置 ----------------------
matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS", "Times New Roman"]
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------------------- 2. 登录系统核心功能 ----------------------
# 密码加密函数（实际应用中应存储加密后的密码）
def encrypt_password(password):
    return sha256(password.encode()).hexdigest()

# 模拟用户数据库（实际应用中应替换为数据库查询）
VALID_USERS = {
    "admin": encrypt_password("admin123"),  # 管理员账号
    "user1": encrypt_password("user123"),   # 普通用户1
    "user2": encrypt_password("user456")    # 普通用户2
}

# 登录状态管理
def check_login_status():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None

def login():
    st.subheader("用户登录")
    
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    
    if st.button("登录", use_container_width=True):
        if username in VALID_USERS and encrypt_password(password) == VALID_USERS[username]:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"登录成功！欢迎回来，{username}")
            st.experimental_rerun()  # 重新加载页面
        else:
            st.error("用户名或密码错误，请重试")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.success("已成功退出登录")
    st.experimental_rerun()

# ---------------------- 3. 自定义CSS ----------------------
st.markdown("""
<style>
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

.login-card {
    max-width: 400px;
    margin: 50px auto;
    padding: 30px;
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

.stButton>button {
    background-color: #3498db !important;
    color: white !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
}
.stButton>button:hover {
    background-color: #2980b9 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- 4. 主应用逻辑 ----------------------
def main_app():
    # 显示登录信息和退出按钮
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("大豆倒伏级别预测系统", anchor=False)
    with col2:
        st.write(f"当前用户: {st.session_state.username}")
        if st.button("退出登录"):
            logout()

    st.markdown("请输入大豆的相关特征参数，系统将预测其倒伏级别并展示特征贡献度。")

    # 加载模型和标准化器
    try:
        model = joblib.load('XGBoost.pkl')
        scaler = joblib.load('data_scaler.pkl')
        # st.success("XGBoost模型和标准化器加载成功!")
    except FileNotFoundError as e:
        st.error(f"文件未找到: {str(e)}! 请确保模型文件和标准化器在当前目录下。")
        st.stop()
    except Exception as e:
        st.error(f"加载失败: {str(e)}")
        st.stop()

    # 特征配置
    feature_ranges = {
        '拉力': {"type": "numerical", "min": 0.0, "max": 999999999.0, "default": 18.8, "step": 0.1},
        '株高': {"type": "numerical", "min": 0.0, "max": 999999999.0, "default": 108.0, "step": 1.0},
        '叶柄长': {"type": "numerical", "min": 0.0, "max": 999999999.0, "default": 17.0, "step": 0.1},
        '节数': {"type": "numerical", "min": 0.0, "max": 999999999.0, "default": 17.0, "step": 1.0}
    }

    feature_name_mapping = {
        '拉力': 'Tensile Force',
        '株高': 'Plant Height',
        '叶柄长': 'Petiole Length',
        '节数': 'Node Number',
        '株高拉力比': 'Height-Force Ratio',
        '叶柄节数比': 'Petiole-Node Ratio'
    }

    model_feature_names = ['拉力', '株高', '叶柄长', '节数', '株高拉力比', '叶柄节数比']
    english_feature_names = [feature_name_mapping[name] for name in model_feature_names]

    lodging_levels = {
        0: {"name": "无倒伏", "description": "作物直立生长，无明显倾斜现象，抗倒伏能力强"},
        1: {"name": "轻度倒伏", "description": "作物倾斜角度小于30°，对产量影响较小，抗倒伏能力较强"},
        2: {"name": "中度倒伏", "description": "作物倾斜角度30°-60°，对产量有一定影响，抗倒伏能力中等"},
        3: {"name": "重度倒伏", "description": "作物倾斜角度大于60°，严重影响产量，抗倒伏能力弱"}
    }

    # 特征输入模块
    with st.container():
        st.markdown('<div class="card"><h3 class="section-title">大豆特征参数</h3>', unsafe_allow_html=True)
        cols = st.columns(2)
        feature_values = {}
        feature_names = list(feature_ranges.keys())

        for idx, (feature, props) in enumerate(feature_ranges.items()):
            with cols[idx % 2]:
                st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 15px;"><div style="width: 100px; padding-right: 10px;">{feature}</div>', unsafe_allow_html=True)
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
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 预测与可视化
    if st.button("预测倒伏级别", type="primary", use_container_width=True, key="predict_btn"):
        # 计算衍生特征
        try:
            if feature_values['拉力'] <= 0:
                st.error("拉力必须大于0，请输入有效的拉力值！")
                st.stop()
            if feature_values['节数'] <= 0:
                st.error("节数必须大于0，请输入有效的节数值！")
                st.stop()
                
            株高拉力比 = feature_values['株高'] / feature_values['拉力']
            叶柄节数比 = feature_values['叶柄长'] / feature_values['节数']
        except Exception as e:
            st.error(f"特征计算错误: {str(e)}")
            st.stop()
        
        # 数据标准化
        try:
            model_input = [
                feature_values['拉力'],
                feature_values['株高'],
                feature_values['叶柄长'],
                feature_values['节数'],
                株高拉力比,
                叶柄节数比
            ]
            input_scaled = scaler.transform([model_input])
            input_data = pd.DataFrame(input_scaled, columns=model_feature_names)
        except Exception as e:
            st.error(f"数据标准化错误: {str(e)}")
            st.stop()
        
        # 模型预测
        try:
            pred_class = model.predict(input_data)[0]
            pred_proba = model.predict_proba(input_data)[0]
            confidence = pred_proba[int(pred_class)] * 100
        except Exception as e:
            st.error(f"模型预测错误: {str(e)}")
            st.stop()

        # SHAP值计算
        try:
            plt.rcParams["font.family"] = ["Arial Unicode MS", "Times New Roman"]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_data)
            base_value = explainer.expected_value
            
            if isinstance(shap_values, list):
                single_shap = shap_values[int(pred_class)][0]
                if isinstance(base_value, list):
                    base_value = base_value[int(pred_class)]
            else:
                single_shap = shap_values[0]
                
        except Exception as e:
            st.warning(f"SHAP值计算失败: {str(e)}，将不显示特征贡献图")
            single_shap = None
            base_value = None

        # 存储结果
        st.session_state.pred_results = {
            "pred_class": pred_class,
            "pred_proba": pred_proba,
            "confidence": confidence,
            "single_shap": single_shap,
            "feature_names": model_feature_names,
            "english_feature_names": english_feature_names,
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
        
        # 输入特征展示
        st.markdown("### 输入特征与计算特征")
        with st.expander("查看详细特征值", expanded=True):
            input_df = pd.DataFrame(list(res["input_features"].items()), columns=["特征", "值"])
            st.dataframe(input_df, use_container_width=True)
            
            derived_df = pd.DataFrame(list(res["derived_features"].items()), columns=["特征", "值"])
            derived_df["值"] = derived_df["值"].apply(lambda x: f"{x:.4f}")
            st.dataframe(derived_df, use_container_width=True)

        # 预测结果展示
        st.markdown("### 倒伏级别预测结果")
        level_info = lodging_levels.get(pred_class, {"name": f"级别{pred_class}", "description": "无详细说明"})
        level_colors = {
            0: "#2ecc71", 1: "#3498db", 2: "#f39c12", 3: "#e74c3c"
        }
        level_color = level_colors.get(pred_class, "#95a5a6")
        
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
        
        # 概率分布展示
        st.markdown("#### 各级别概率分布")
        prob_data = []
        for i, prob in enumerate(res["pred_proba"]):
            level_name = lodging_levels.get(i, {"name": f"级别{i}"})["name"]
            prob_data.append({
                "倒伏级别": f"{i} ({level_name})",
                "概率": f"{prob*100:.2f}%"
            })
        prob_df = pd.DataFrame(prob_data)
        st.dataframe(prob_df, use_container_width=True)

        # SHAP图展示
        if res["single_shap"] is not None and res["base_value"] is not None:
            st.markdown("### SHAP特征贡献图 (Feature Contribution)")
            st.markdown("蓝色 = 降低倒伏风险，红色 = 增加倒伏风险，长度 = 贡献程度")
            
            shap_exp = shap.Explanation(
                values=res['single_shap'],
                base_values=res['base_value'],
                data=res['feature_values'],
                feature_names=res['english_feature_names']
            )

            plt.figure(figsize=(12, 8))
            shap.plots.waterfall(shap_exp, max_display=10, show=False)
            
            for ax in plt.gcf().axes:
                ax.set_title(ax.get_title(), fontfamily="Times New Roman", fontsize=12)
                ax.set_xlabel(ax.get_xlabel(), fontfamily="Times New Roman", fontsize=10)
                ax.set_ylabel(ax.get_ylabel(), fontfamily="Times New Roman", fontsize=10)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontfamily("Times New Roman")
                    label.set_fontsize(9)
            
            plt.tight_layout()
            st.pyplot(plt.gcf())

            # SHAP值表格
            if st.checkbox("显示所有特征的SHAP值 (Show all SHAP values)", key="show_shap"):
                shap_df = pd.DataFrame({
                    "特征 (Feature)": [f"{cn} ({en})" for cn, en in zip(res['feature_names'], res['english_feature_names'])],
                    "数值 (Value)": [round(v, 4) for v in res['feature_values']],
                    "SHAP值 (贡献度)": res['single_shap'].round(4)
                })
                shap_df["绝对贡献度 (Absolute Contribution)"] = shap_df["SHAP值 (贡献度)"].abs()
                shap_df_sorted = shap_df.sort_values("绝对贡献度 (Absolute Contribution)", ascending=False).drop("绝对贡献度 (Absolute Contribution)", axis=1)
                st.dataframe(shap_df_sorted, use_container_width=True)

# ---------------------- 5. 程序入口 ----------------------
if __name__ == "__main__":
    check_login_status()
    
    if not st.session_state.logged_in:
        # 显示登录界面
        st.markdown('<div class="card login-card">', unsafe_allow_html=True)
        login()
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # 显示主应用
        main_app()
