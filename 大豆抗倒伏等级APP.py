# import streamlit as st
# import numpy as np
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt
# import joblib
# import matplotlib
# from hashlib import sha256  # 用于密码加密存储

# # ---------------------- 1. 基础配置 ----------------------
# matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS", "Times New Roman"]
# matplotlib.rcParams['axes.unicode_minus'] = False

# # ---------------------- 2. 登录系统核心功能 ----------------------
# # 密码加密函数（实际应用中应存储加密后的密码）
# def encrypt_password(password):
#     return sha256(password.encode()).hexdigest()

# # 模拟用户数据库（实际应用中应替换为数据库查询）
# VALID_USERS = {
#     "admin": encrypt_password("admin123"),  # 管理员账号
#     "user1": encrypt_password("user123"),   # 普通用户1
#     "user2": encrypt_password("user456")    # 普通用户2
# }

# # 登录状态管理
# def check_login_status():
#     if "logged_in" not in st.session_state:
#         st.session_state.logged_in = False
#         st.session_state.username = None

# def login():
#     st.subheader("用户登录")
    
#     username = st.text_input("用户名")
#     password = st.text_input("密码", type="password")
    
#     if st.button("登录", use_container_width=True):
#         if username in VALID_USERS and encrypt_password(password) == VALID_USERS[username]:
#             st.session_state.logged_in = True
#             st.session_state.username = username
#             st.success(f"登录成功！欢迎回来，{username}")
#             st.experimental_rerun()  # 重新加载页面
#         else:
#             st.error("用户名或密码错误，请重试")

# def logout():
#     st.session_state.logged_in = False
#     st.session_state.username = None
#     st.success("已成功退出登录")
#     st.experimental_rerun()

# # ---------------------- 3. 自定义CSS ----------------------
# st.markdown("""
# <style>
# * {
#     font-family: "SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Helvetica Neue", Arial, sans-serif !important;
# }

# body {
#     background-color: #f5f7fa;
# }

# .card {
#     background-color: white;
#     border-radius: 8px;
#     box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
#     padding: 20px;
#     margin-bottom: 20px;
# }

# .section-title {
#     font-size: 18px;
#     font-weight: bold;
#     color: #2c3e50;
#     border-bottom: 2px solid #3498db;
#     padding-bottom: 10px;
#     margin-bottom: 15px;
# }

# .login-card {
#     max-width: 400px;
#     margin: 50px auto;
#     padding: 30px;
# }

# .result-card {
#     border-radius: 8px;
#     padding: 15px;
#     margin: 10px 0;
#     color: white;
#     font-weight: bold;
# }

# .confidence-bar {
#     height: 20px;
#     border-radius: 10px;
#     margin: 5px 0;
#     background-color: #e0e0e0;
#     overflow: hidden;
# }

# .confidence-fill {
#     height: 100%;
# }

# .stButton>button {
#     background-color: #3498db !important;
#     color: white !important;
#     border-radius: 6px !important;
#     padding: 8px 16px !important;
# }
# .stButton>button:hover {
#     background-color: #2980b9 !important;
# }
# </style>
# """, unsafe_allow_html=True)

# # ---------------------- 4. 主应用逻辑 ----------------------
# def main_app():
#     # 显示登录信息和退出按钮
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         st.title("大豆倒伏级别预测系统", anchor=False)
#     with col2:
#         st.write(f"当前用户: {st.session_state.username}")
#         if st.button("退出登录"):
#             logout()

#     st.markdown("请输入大豆的相关特征参数，系统将预测其倒伏级别并展示特征贡献度。")

#     # 加载模型和标准化器
#     try:
#         model = joblib.load('XGBoost.pkl')
#         scaler = joblib.load('data_scaler.pkl')
#         # st.success("XGBoost模型和标准化器加载成功!")
#     except FileNotFoundError as e:
#         st.error(f"文件未找到: {str(e)}! 请确保模型文件和标准化器在当前目录下。")
#         st.stop()
#     except Exception as e:
#         st.error(f"加载失败: {str(e)}")
#         st.stop()

#     # 特征配置
#     feature_ranges = {
#         '拉力': {"type": "numerical", "min": 0.0, "max": 999999999.0, "default": 18.8, "step": 0.1},
#         '株高': {"type": "numerical", "min": 0.0, "max": 999999999.0, "default": 108.0, "step": 1.0},
#         '叶柄长': {"type": "numerical", "min": 0.0, "max": 999999999.0, "default": 17.0, "step": 0.1},
#         '节数': {"type": "numerical", "min": 0.0, "max": 999999999.0, "default": 17.0, "step": 1.0}
#     }

#     feature_name_mapping = {
#         '拉力': 'Tensile Force',
#         '株高': 'Plant Height',
#         '叶柄长': 'Petiole Length',
#         '节数': 'Node Number',
#         '株高拉力比': 'Height-Force Ratio',
#         '叶柄节数比': 'Petiole-Node Ratio'
#     }

#     model_feature_names = ['拉力', '株高', '叶柄长', '节数', '株高拉力比', '叶柄节数比']
#     english_feature_names = [feature_name_mapping[name] for name in model_feature_names]

#     lodging_levels = {
#         0: {"name": "无倒伏", "description": "作物直立生长，无明显倾斜现象，抗倒伏能力强"},
#         1: {"name": "轻度倒伏", "description": "作物倾斜角度小于30°，对产量影响较小，抗倒伏能力较强"},
#         2: {"name": "中度倒伏", "description": "作物倾斜角度30°-60°，对产量有一定影响，抗倒伏能力中等"},
#         3: {"name": "重度倒伏", "description": "作物倾斜角度大于60°，严重影响产量，抗倒伏能力弱"}
#     }

#     # 特征输入模块
#     with st.container():
#         st.markdown('<div class="card"><h3 class="section-title">大豆特征参数</h3>', unsafe_allow_html=True)
#         cols = st.columns(2)
#         feature_values = {}
#         feature_names = list(feature_ranges.keys())

#         for idx, (feature, props) in enumerate(feature_ranges.items()):
#             with cols[idx % 2]:
#                 st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 15px;"><div style="width: 100px; padding-right: 10px;">{feature}</div>', unsafe_allow_html=True)
#                 if props["type"] == "numerical":
#                     value = st.number_input(
#                         feature,
#                         min_value=float(props["min"]),
#                         max_value=float(props["max"]),
#                         value=float(props["default"]),
#                         step=props["step"],
#                         format="%.1f" if props["step"] < 1 else "%.0f",
#                         label_visibility="collapsed"
#                     )
#                 feature_values[feature] = value
#                 st.markdown('</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)

#     # 预测与可视化
#     if st.button("预测倒伏级别", type="primary", use_container_width=True, key="predict_btn"):
#         # 计算衍生特征
#         try:
#             if feature_values['拉力'] <= 0:
#                 st.error("拉力必须大于0，请输入有效的拉力值！")
#                 st.stop()
#             if feature_values['节数'] <= 0:
#                 st.error("节数必须大于0，请输入有效的节数值！")
#                 st.stop()
                
#             株高拉力比 = feature_values['株高'] / feature_values['拉力']
#             叶柄节数比 = feature_values['叶柄长'] / feature_values['节数']
#         except Exception as e:
#             st.error(f"特征计算错误: {str(e)}")
#             st.stop()
        
#         # 数据标准化
#         try:
#             model_input = [
#                 feature_values['拉力'],
#                 feature_values['株高'],
#                 feature_values['叶柄长'],
#                 feature_values['节数'],
#                 株高拉力比,
#                 叶柄节数比
#             ]
#             input_scaled = scaler.transform([model_input])
#             input_data = pd.DataFrame(input_scaled, columns=model_feature_names)
#         except Exception as e:
#             st.error(f"数据标准化错误: {str(e)}")
#             st.stop()
        
#         # 模型预测
#         try:
#             pred_class = model.predict(input_data)[0]
#             pred_proba = model.predict_proba(input_data)[0]
#             confidence = pred_proba[int(pred_class)] * 100
#         except Exception as e:
#             st.error(f"模型预测错误: {str(e)}")
#             st.stop()

#         # SHAP值计算
#         try:
#             plt.rcParams["font.family"] = ["Arial Unicode MS", "Times New Roman"]
#             explainer = shap.TreeExplainer(model)
#             shap_values = explainer.shap_values(input_data)
#             base_value = explainer.expected_value
            
#             if isinstance(shap_values, list):
#                 single_shap = shap_values[int(pred_class)][0]
#                 if isinstance(base_value, list):
#                     base_value = base_value[int(pred_class)]
#             else:
#                 single_shap = shap_values[0]
                
#         except Exception as e:
#             st.warning(f"SHAP值计算失败: {str(e)}，将不显示特征贡献图")
#             single_shap = None
#             base_value = None

#         # 存储结果
#         st.session_state.pred_results = {
#             "pred_class": pred_class,
#             "pred_proba": pred_proba,
#             "confidence": confidence,
#             "single_shap": single_shap,
#             "feature_names": model_feature_names,
#             "english_feature_names": english_feature_names,
#             "feature_values": model_input,
#             "base_value": base_value,
#             "input_features": feature_values,
#             "derived_features": {
#                 "株高拉力比": 株高拉力比,
#                 "叶柄节数比": 叶柄节数比
#             }
#         }

#     # 显示预测结果
#     if "pred_results" in st.session_state:
#         res = st.session_state.pred_results
#         pred_class = int(res["pred_class"])
        
#         # 输入特征展示
#         st.markdown("### 输入特征与计算特征")
#         with st.expander("查看详细特征值", expanded=True):
#             input_df = pd.DataFrame(list(res["input_features"].items()), columns=["特征", "值"])
#             st.dataframe(input_df, use_container_width=True)
            
#             derived_df = pd.DataFrame(list(res["derived_features"].items()), columns=["特征", "值"])
#             derived_df["值"] = derived_df["值"].apply(lambda x: f"{x:.4f}")
#             st.dataframe(derived_df, use_container_width=True)

#         # 预测结果展示
#         st.markdown("### 倒伏级别预测结果")
#         level_info = lodging_levels.get(pred_class, {"name": f"级别{pred_class}", "description": "无详细说明"})
#         level_colors = {
#             0: "#2ecc71", 1: "#3498db", 2: "#f39c12", 3: "#e74c3c"
#         }
#         level_color = level_colors.get(pred_class, "#95a5a6")
        
#         st.markdown(f"""
#         <div class="result-card" style="background-color: {level_color};">
#             预测倒伏级别: {pred_class} ({level_info['name']})
#         </div>
#         <div style="margin: 10px 0;">
#             <p><strong>级别描述:</strong> {level_info['description']}</p>
#             <p><strong>预测置信度:</strong> {res['confidence']:.2f}%</p>
#             <div class="confidence-bar">
#                 <div class="confidence-fill" style="width: {res['confidence']}%; background-color: {level_color};"></div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # 概率分布展示
#         st.markdown("#### 各级别概率分布")
#         prob_data = []
#         for i, prob in enumerate(res["pred_proba"]):
#             level_name = lodging_levels.get(i, {"name": f"级别{i}"})["name"]
#             prob_data.append({
#                 "倒伏级别": f"{i} ({level_name})",
#                 "概率": f"{prob*100:.2f}%"
#             })
#         prob_df = pd.DataFrame(prob_data)
#         st.dataframe(prob_df, use_container_width=True)

#         # SHAP图展示
#         if res["single_shap"] is not None and res["base_value"] is not None:
#             st.markdown("### SHAP特征贡献图 (Feature Contribution)")
#             st.markdown("蓝色 = 降低倒伏风险，红色 = 增加倒伏风险，长度 = 贡献程度")
            
#             shap_exp = shap.Explanation(
#                 values=res['single_shap'],
#                 base_values=res['base_value'],
#                 data=res['feature_values'],
#                 feature_names=res['english_feature_names']
#             )

#             plt.figure(figsize=(12, 8))
#             shap.plots.waterfall(shap_exp, max_display=10, show=False)
            
#             for ax in plt.gcf().axes:
#                 ax.set_title(ax.get_title(), fontfamily="Times New Roman", fontsize=12)
#                 ax.set_xlabel(ax.get_xlabel(), fontfamily="Times New Roman", fontsize=10)
#                 ax.set_ylabel(ax.get_ylabel(), fontfamily="Times New Roman", fontsize=10)
#                 for label in ax.get_xticklabels() + ax.get_yticklabels():
#                     label.set_fontfamily("Times New Roman")
#                     label.set_fontsize(9)
            
#             plt.tight_layout()
#             st.pyplot(plt.gcf())

#             # SHAP值表格
#             if st.checkbox("显示所有特征的SHAP值 (Show all SHAP values)", key="show_shap"):
#                 shap_df = pd.DataFrame({
#                     "特征 (Feature)": [f"{cn} ({en})" for cn, en in zip(res['feature_names'], res['english_feature_names'])],
#                     "数值 (Value)": [round(v, 4) for v in res['feature_values']],
#                     "SHAP值 (贡献度)": res['single_shap'].round(4)
#                 })
#                 shap_df["绝对贡献度 (Absolute Contribution)"] = shap_df["SHAP值 (贡献度)"].abs()
#                 shap_df_sorted = shap_df.sort_values("绝对贡献度 (Absolute Contribution)", ascending=False).drop("绝对贡献度 (Absolute Contribution)", axis=1)
#                 st.dataframe(shap_df_sorted, use_container_width=True)

# # ---------------------- 5. 程序入口 ----------------------
# if __name__ == "__main__":
#     check_login_status()
    
#     if not st.session_state.logged_in:
#         # 显示登录界面
#         st.markdown('<div class="card login-card">', unsafe_allow_html=True)
#         login()
#         st.markdown('</div>', unsafe_allow_html=True)
#     else:
#         # 显示主应用



# ###########################################################################################################################################################################################################################

# import streamlit as st
# import numpy as np
# import pandas as pd
# import shap
# import matplotlib.pyplot as plt
# import joblib
# import matplotlib
# import sqlite3
# from hashlib import sha256
# import re
# from datetime import datetime

# # ---------------------- 1. 基础配置 ----------------------
# matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS", "Times New Roman"]
# matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# # ---------------------- 2. 数据库操作（用户管理） ----------------------
# def init_db():
#     """初始化SQLite数据库，创建用户表"""
#     conn = sqlite3.connect('user_db.db')
#     c = conn.cursor()
#     # 创建用户表：用户名（主键）、加密密码、注册时间
#     c.execute('''CREATE TABLE IF NOT EXISTS users
#                  (username TEXT PRIMARY KEY NOT NULL,
#                   password TEXT NOT NULL,
#                   create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
#     conn.commit()
#     conn.close()

# def encrypt_password(password):
#     """密码加密（SHA-256 + 盐值增强安全性）"""
#     salt = "soybean_lodging_system_2024_salt"  # 自定义盐值，实际使用时建议修改
#     return sha256((password + salt).encode()).hexdigest()

# def add_user(username, password):
#     """新增用户（注册功能）"""
#     conn = sqlite3.connect('user_db.db')
#     c = conn.cursor()
#     try:
#         encrypted_pwd = encrypt_password(password)
#         create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         c.execute("INSERT INTO users (username, password, create_time) VALUES (?, ?, ?)", 
#                  (username, encrypted_pwd, create_time))
#         conn.commit()
#         conn.close()
#         return True  # 注册成功
#     except sqlite3.IntegrityError:
#         conn.close()
#         return False  # 用户名已存在
#     except Exception as e:
#         conn.close()
#         print(f"注册错误: {e}")
#         return False

# def verify_user(username, password):
#     """验证用户（登录功能）"""
#     conn = sqlite3.connect('user_db.db')
#     c = conn.cursor()
#     encrypted_pwd = encrypt_password(password)
#     c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, encrypted_pwd))
#     result = c.fetchone()
#     conn.close()
#     return result is not None  # 验证通过返回True

# def update_password(username, old_password, new_password):
#     """修改密码功能"""
#     # 先验证旧密码
#     if not verify_user(username, old_password):
#         return False, "旧密码验证失败"
    
#     conn = sqlite3.connect('user_db.db')
#     c = conn.cursor()
#     try:
#         new_encrypted_pwd = encrypt_password(new_password)
#         c.execute("UPDATE users SET password = ? WHERE username = ?", (new_encrypted_pwd, username))
#         conn.commit()
#         conn.close()
#         return True, "密码修改成功"
#     except Exception as e:
#         conn.close()
#         return False, f"修改失败: {str(e)}"

# # 初始化数据库（首次运行自动创建）
# init_db()

# # ---------------------- 3. 会话状态管理 ----------------------
# def init_session_state():
#     """初始化会话状态变量"""
#     if "logged_in" not in st.session_state:
#         st.session_state.logged_in = False
#         st.session_state.username = None
#     if "current_page" not in st.session_state:
#         st.session_state.current_page = "login"  # 页面状态：login/register/change_pwd/main
#     if "pred_results" not in st.session_state:
#         st.session_state.pred_results = None  # 存储预测结果

# # 页面切换函数
# def go_to_login():
#     st.session_state.current_page = "login"
# def go_to_register():
#     st.session_state.current_page = "register"
# def go_to_change_pwd():
#     st.session_state.current_page = "change_pwd"
# def go_to_main():
#     st.session_state.current_page = "main"

# # ---------------------- 4. 自定义CSS样式 ----------------------
# st.markdown("""
# <style>
# * {
#     font-family: "SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Helvetica Neue", Arial, sans-serif !important;
# }

# body {
#     background-color: #f5f7fa;
# }

# .card {
#     background-color: white;
#     border-radius: 8px;
#     box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
#     padding: 20px;
#     margin-bottom: 20px;
# }

# .auth-card {
#     max-width: 450px;
#     margin: 30px auto;
#     padding: 30px;
# }

# .section-title {
#     font-size: 18px;
#     font-weight: bold;
#     color: #2c3e50;
#     border-bottom: 2px solid #3498db;
#     padding-bottom: 10px;
#     margin-bottom: 20px;
#     text-align: center;
# }

# .result-card {
#     border-radius: 8px;
#     padding: 15px;
#     margin: 10px 0;
#     color: white;
#     font-weight: bold;
# }

# .confidence-bar {
#     height: 20px;
#     border-radius: 10px;
#     margin: 5px 0;
#     background-color: #e0e0e0;
#     overflow: hidden;
# }

# .confidence-fill {
#     height: 100%;
# }

# /* 按钮样式 */
# .stButton>button {
#     background-color: #3498db !important;
#     color: white !important;
#     border-radius: 6px !important;
#     padding: 8px 16px !important;
#     margin: 5px 0;
# }
# .stButton>button:hover {
#     background-color: #2980b9 !important;
# }
# .stButton>button.secondary {
#     background-color: #95a5a6 !important;
# }
# .stButton>button.secondary:hover {
#     background-color: #7f8c8d !important;
# }

# /* 输入框样式 */
# .stTextInput, .stNumberInput {
#     margin-bottom: 15px;
# }

# /* 链接样式 */
# .link {
#     color: #3498db;
#     text-decoration: underline;
#     cursor: pointer;
#     text-align: center;
#     margin-top: 15px;
# }
# .link:hover {
#     color: #2980b9;
# }
# </style>
# """, unsafe_allow_html=True)

# # ---------------------- 5. 认证页面（登录/注册/修改密码） ----------------------
# def login_page():
#     """登录页面"""
#     st.markdown('<div class="card auth-card">', unsafe_allow_html=True)
#     st.markdown('<h3 class="section-title">用户登录</h3>', unsafe_allow_html=True)
    
#     username = st.text_input("用户名", placeholder="请输入用户名")
#     password = st.text_input("密码", type="password", placeholder="请输入密码")
    
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("登录", use_container_width=True):
#             if not username or not password:
#                 st.error("用户名和密码不能为空")
#                 return
#             if verify_user(username, password):
#                 st.session_state.logged_in = True
#                 st.session_state.username = username
#                 go_to_main()
#                 st.success("登录成功！正在跳转...")
#                 st.rerun()
#             else:
#                 st.error("用户名或密码错误")
#     with col2:
#         st.button("注册账号", use_container_width=True, on_click=go_to_register, type="secondary")
    
#     # 忘记密码链接
#     st.markdown("""
#     <div class="link" onclick="window.parent.pythonFunction('change_pwd')">
#         忘记密码？修改密码
#     </div>
#     """, unsafe_allow_html=True)
#     st.markdown('</div>', unsafe_allow_html=True)

# def register_page():
#     """注册页面"""
#     st.markdown('<div class="card auth-card">', unsafe_allow_html=True)
#     st.markdown('<h3 class="section-title">用户注册</h3>', unsafe_allow_html=True)
    
#     username = st.text_input("用户名", placeholder="请设置用户名（3-20个字符）")
#     password = st.text_input("密码", type="password", placeholder="请设置密码（6-20个字符，含字母和数字）")
#     confirm_pwd = st.text_input("确认密码", type="password", placeholder="请再次输入密码")
    
#     # 密码强度验证
#     def validate_password(pwd):
#         if len(pwd) < 6 or len(pwd) > 20:
#             return False, "密码长度需在6-20个字符之间"
#         if not re.search(r'[a-zA-Z]', pwd) or not re.search(r'[0-9]', pwd):
#             return False, "密码需同时包含字母和数字"
#         return True, ""
    
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("注册", use_container_width=True):
#             if not username or not password or not confirm_pwd:
#                 st.error("所有字段不能为空")
#                 return
#             if len(username) < 3 or len(username) > 20:
#                 st.error("用户名长度需在3-20个字符之间")
#                 return
#             if password != confirm_pwd:
#                 st.error("两次输入的密码不一致")
#                 return
#             # 验证密码强度
#             pwd_valid, pwd_msg = validate_password(password)
#             if not pwd_valid:
#                 st.error(pwd_msg)
#                 return
#             # 新增用户
#             if add_user(username, password):
#                 st.success("注册成功！请登录")
#                 go_to_login()
#                 st.rerun()
#             else:
#                 st.error("用户名已存在，请更换")
#     with col2:
#         st.button("返回登录", use_container_width=True, on_click=go_to_login, type="secondary")
    
#     st.markdown('</div>', unsafe_allow_html=True)

# def change_password_page():
#     """修改密码页面"""
#     st.markdown('<div class="card auth-card">', unsafe_allow_html=True)
#     st.markdown('<h3 class="section-title">修改密码</h3>', unsafe_allow_html=True)
    
#     username = st.text_input("用户名", placeholder="请输入您的用户名")
#     old_password = st.text_input("旧密码", type="password", placeholder="请输入旧密码")
#     new_password = st.text_input("新密码", type="password", placeholder="请输入新密码（6-20个字符，含字母和数字）")
#     confirm_new_pwd = st.text_input("确认新密码", type="password", placeholder="请再次输入新密码")
    
#     # 密码强度验证（复用注册时的函数）
#     def validate_password(pwd):
#         if len(pwd) < 6 or len(pwd) > 20:
#             return False, "密码长度需在6-20个字符之间"
#         if not re.search(r'[a-zA-Z]', pwd) or not re.search(r'[0-9]', pwd):
#             return False, "密码需同时包含字母和数字"
#         return True, ""
    
#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("确认修改", use_container_width=True):
#             if not username or not old_password or not new_password or not confirm_new_pwd:
#                 st.error("所有字段不能为空")
#                 return
#             if new_password != confirm_new_pwd:
#                 st.error("两次输入的新密码不一致")
#                 return
#             # 验证新密码强度
#             pwd_valid, pwd_msg = validate_password(new_password)
#             if not pwd_valid:
#                 st.error(pwd_msg)
#                 return
#             # 执行修改
#             success, msg = update_password(username, old_password, new_password)
#             if success:
#                 st.success(msg)
#                 go_to_login()
#                 st.rerun()
#             else:
#                 st.error(msg)
#     with col2:
#         st.button("返回登录", use_container_width=True, on_click=go_to_login, type="secondary")
    
#     st.markdown('</div>', unsafe_allow_html=True)

# # ---------------------- 6. 主应用功能（大豆倒伏预测） ----------------------
# def main_app():
#     # 顶部导航栏：标题 + 修改密码 + 退出登录
#     col1, col2, col3 = st.columns([3, 1, 1])
#     with col1:
#         st.title("大豆倒伏级别预测系统", anchor=False)
#     with col2:
#         if st.button("修改密码", use_container_width=True, type="secondary"):
#             go_to_change_pwd()
#             st.rerun()
#     with col3:
#         if st.button("退出登录", use_container_width=True):
#             st.session_state.logged_in = False
#             st.session_state.username = None
#             st.session_state.pred_results = None
#             go_to_login()
#             st.rerun()
    
#     st.markdown(f"欢迎回来，{st.session_state.username}！请输入大豆的相关特征参数，系统将预测其倒伏级别并展示特征贡献度。")

#     # 加载模型和标准化器
#     try:
#         model = joblib.load('XGBoost.pkl')
#         scaler = joblib.load('data_scaler.pkl')
#     except FileNotFoundError as e:
#         st.error(f"文件未找到: {str(e)}! 请确保模型文件和标准化器在当前目录下。")
#         st.stop()
#     except Exception as e:
#         st.error(f"加载失败: {str(e)}")
#         st.stop()

#     # 特征配置
#     feature_ranges = {
#         '拉力': {"type": "numerical", "min": 0.1, "max": 999999999.0, "default": 18.8, "step": 0.1},
#         '株高': {"type": "numerical", "min": 0.1, "max": 999999999.0, "default": 108.0, "step": 1.0},
#         '叶柄长': {"type": "numerical", "min": 0.1, "max": 999999999.0, "default": 17.0, "step": 0.1},
#         '节数': {"type": "numerical", "min": 1.0, "max": 999999999.0, "default": 17.0, "step": 1.0}
#     }

#     feature_name_mapping = {
#         '拉力': 'Tensile Force',
#         '株高': 'Plant Height',
#         '叶柄长': 'Petiole Length',
#         '节数': 'Node Number',
#         '株高拉力比': 'Height-Force Ratio',
#         '叶柄节数比': 'Petiole-Node Ratio'
#     }

#     model_feature_names = ['拉力', '株高', '叶柄长', '节数', '株高拉力比', '叶柄节数比']
#     english_feature_names = [feature_name_mapping[name] for name in model_feature_names]

#     lodging_levels = {
#         0: {"name": "无倒伏", "description": "作物直立生长，无明显倾斜现象，抗倒伏能力强"},
#         1: {"name": "轻度倒伏", "description": "作物倾斜角度小于30°，对产量影响较小，抗倒伏能力较强"},
#         2: {"name": "中度倒伏", "description": "作物倾斜角度30°-60°，对产量有一定影响，抗倒伏能力中等"},
#         3: {"name": "重度倒伏", "description": "作物倾斜角度大于60°，严重影响产量，抗倒伏能力弱"}
#     }

#     # 特征输入模块
#     with st.container():
#         st.markdown('<div class="card"><h3 class="section-title">大豆特征参数</h3>', unsafe_allow_html=True)
#         cols = st.columns(2)
#         feature_values = {}
#         feature_names = list(feature_ranges.keys())

#         for idx, (feature, props) in enumerate(feature_ranges.items()):
#             with cols[idx % 2]:
#                 st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 15px;"><div style="width: 100px; padding-right: 10px;">{feature}</div>', unsafe_allow_html=True)
#                 value = st.number_input(
#                     feature,
#                     min_value=float(props["min"]),
#                     max_value=float(props["max"]),
#                     value=float(props["default"]),
#                     step=props["step"],
#                     format="%.1f" if props["step"] < 1 else "%.0f",
#                     label_visibility="collapsed"
#                 )
#                 feature_values[feature] = value
#                 st.markdown('</div>', unsafe_allow_html=True)
#         st.markdown('</div>', unsafe_allow_html=True)

#     # 预测与可视化
#     if st.button("预测倒伏级别", type="primary", use_container_width=True, key="predict_btn"):
#         # 计算衍生特征
#         try:
#             株高拉力比 = feature_values['株高'] / feature_values['拉力']
#             叶柄节数比 = feature_values['叶柄长'] / feature_values['节数']
#         except Exception as e:
#             st.error(f"特征计算错误: {str(e)}")
#             st.stop()
        
#         # 数据标准化
#         try:
#             model_input = [
#                 feature_values['拉力'],
#                 feature_values['株高'],
#                 feature_values['叶柄长'],
#                 feature_values['节数'],
#                 株高拉力比,
#                 叶柄节数比
#             ]
#             input_scaled = scaler.transform([model_input])
#             input_data = pd.DataFrame(input_scaled, columns=model_feature_names)
#         except Exception as e:
#             st.error(f"数据标准化错误: {str(e)}")
#             st.stop()
        
#         # 模型预测
#         try:
#             pred_class = model.predict(input_data)[0]
#             pred_proba = model.predict_proba(input_data)[0]
#             confidence = pred_proba[int(pred_class)] * 100
#         except Exception as e:
#             st.error(f"模型预测错误: {str(e)}")
#             st.stop()

#         # SHAP值计算
#         try:
#             plt.rcParams["font.family"] = ["Arial Unicode MS", "Times New Roman"]
#             explainer = shap.TreeExplainer(model)
#             shap_values = explainer.shap_values(input_data)
#             base_value = explainer.expected_value
            
#             if isinstance(shap_values, list):
#                 single_shap = shap_values[int(pred_class)][0]
#                 if isinstance(base_value, list):
#                     base_value = base_value[int(pred_class)]
#             else:
#                 single_shap = shap_values[0]
                
#         except Exception as e:
#             st.warning(f"SHAP值计算失败: {str(e)}，将不显示特征贡献图")
#             single_shap = None
#             base_value = None

#         # 存储结果到会话状态
#         st.session_state.pred_results = {
#             "pred_class": pred_class,
#             "pred_proba": pred_proba,
#             "confidence": confidence,
#             "single_shap": single_shap,
#             "feature_names": model_feature_names,
#             "english_feature_names": english_feature_names,
#             "feature_values": model_input,
#             "base_value": base_value,
#             "input_features": feature_values,
#             "derived_features": {
#                 "株高拉力比": 株高拉力比,
#                 "叶柄节数比": 叶柄节数比
#             }
#         }

#     # 显示预测结果
#     if st.session_state.pred_results is not None:
#         res = st.session_state.pred_results
#         pred_class = int(res["pred_class"])
        
#         # 输入特征展示
#         st.markdown("### 输入特征与计算特征")
#         with st.expander("查看详细特征值", expanded=True):
#             input_df = pd.DataFrame(list(res["input_features"].items()), columns=["特征", "值"])
#             st.dataframe(input_df, use_container_width=True)
            
#             derived_df = pd.DataFrame(list(res["derived_features"].items()), columns=["特征", "值"])
#             derived_df["值"] = derived_df["值"].apply(lambda x: f"{x:.4f}")
#             st.dataframe(derived_df, use_container_width=True)

#         # 预测结果展示
#         st.markdown("### 倒伏级别预测结果")
#         level_info = lodging_levels.get(pred_class, {"name": f"级别{pred_class}", "description": "无详细说明"})
#         level_colors = {
#             0: "#2ecc71",  # 绿色-无倒伏
#             1: "#3498db",  # 蓝色-轻度
#             2: "#f39c12",  # 橙色-中度
#             3: "#e74c3c"   # 红色-重度
#         }
#         level_color = level_colors.get(pred_class, "#95a5a6")
        
#         st.markdown(f"""
#         <div class="result-card" style="background-color: {level_color};">
#             预测倒伏级别: {pred_class} ({level_info['name']})
#         </div>
#         <div style="margin: 10px 0;">
#             <p><strong>级别描述:</strong> {level_info['description']}</p>
#             <p><strong>预测置信度:</strong> {res['confidence']:.2f}%</p>
#             <div class="confidence-bar">
#                 <div class="confidence-fill" style="width: {res['confidence']}%; background-color: {level_color};"></div>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # 概率分布展示
#         st.markdown("#### 各级别概率分布")
#         prob_data = []
#         for i, prob in enumerate(res["pred_proba"]):
#             level_name = lodging_levels.get(i, {"name": f"级别{i}"})["name"]
#             prob_data.append({
#                 "倒伏级别": f"{i} ({level_name})",
#                 "概率": f"{prob*100:.2f}%"
#             })
#         prob_df = pd.DataFrame(prob_data)
#         st.dataframe(prob_df, use_container_width=True)

#         # SHAP图展示
#         if res["single_shap"] is not None and res["base_value"] is not None:
#             st.markdown("### SHAP特征贡献图 (Feature Contribution)")
#             st.markdown("蓝色 = 降低倒伏风险，红色 = 增加倒伏风险，长度 = 贡献程度")
            
#             shap_exp = shap.Explanation(
#                 values=res['single_shap'],
#                 base_values=res['base_value'],
#                 data=res['feature_values'],
#                 feature_names=res['english_feature_names']
#             )

#             plt.figure(figsize=(12, 8))
#             shap.plots.waterfall(shap_exp, max_display=10, show=False)
            
#             # 设置图表字体为英文
#             for ax in plt.gcf().axes:
#                 ax.set_title(ax.get_title(), fontfamily="Times New Roman", fontsize=12)
#                 ax.set_xlabel(ax.get_xlabel(), fontfamily="Times New Roman", fontsize=10)
#                 ax.set_ylabel(ax.get_ylabel(), fontfamily="Times New Roman", fontsize=10)
#                 for label in ax.get_xticklabels() + ax.get_yticklabels():
#                     label.set_fontfamily("Times New Roman")
#                     label.set_fontsize(9)
            
#             plt.tight_layout()
#             st.pyplot(plt.gcf())

#             # SHAP值表格（中英文对照）
#             if st.checkbox("显示所有特征的SHAP值 (Show all SHAP values)", key="show_shap"):
#                 shap_df = pd.DataFrame({
#                     "特征 (Feature)": [f"{cn} ({en})" for cn, en in zip(res['feature_names'], res['english_feature_names'])],
#                     "数值 (Value)": [round(v, 4) for v in res['feature_values']],
#                     "SHAP值 (贡献度)": res['single_shap'].round(4)
#                 })
#                 shap_df["绝对贡献度 (Absolute Contribution)"] = shap_df["SHAP值 (贡献度)"].abs()
#                 shap_df_sorted = shap_df.sort_values("绝对贡献度 (Absolute Contribution)", ascending=False).drop("绝对贡献度 (Absolute Contribution)", axis=1)
#                 st.dataframe(shap_df_sorted, use_container_width=True)

# # ---------------------- 7. 程序入口 ----------------------
# if __name__ == "__main__":
#     init_session_state()
    
#     # 根据当前页面状态显示对应内容
#     if st.session_state.current_page == "login":
#         login_page()
#     elif st.session_state.current_page == "register":
#         register_page()
#     elif st.session_state.current_page == "change_pwd":
#         change_password_page()
#     elif st.session_state.current_page == "main":
#         main_app()
# #         main_app()




# ###########################################################################################################################################################################################################################

import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import matplotlib
import sqlite3
from hashlib import sha256
import re
from datetime import datetime

# ---------------------- 1. 基础配置 ----------------------
matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS", "Times New Roman"]
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------------------- 2. 数据库操作（用户管理+权限控制） ----------------------
def init_db():
    """初始化数据库，增加权限字段（admin/user）"""
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    # 创建用户表：用户名、加密密码、权限(admin/user)、注册时间
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY NOT NULL,
                  password TEXT NOT NULL,
                  role TEXT NOT NULL DEFAULT 'user',  -- 权限：admin或user
                  create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # 初始化管理员账号（首次运行自动创建，可手动修改密码）
    admin_pwd = encrypt_password("admin123")  # 初始管理员密码
    try:
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                 ("admin", admin_pwd, "admin"))
    except sqlite3.IntegrityError:
        pass  # 管理员已存在则跳过
    conn.commit()
    conn.close()

def encrypt_password(password):
    """密码加密（SHA-256 + 盐值）"""
    salt = "soybean_lodging_system_salt_2024"
    return sha256((password + salt).encode()).hexdigest()

def get_user_role(username):
    """获取用户权限（admin/user）"""
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    c.execute("SELECT role FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def add_user(username, password, role="user"):
    """新增用户（仅管理员可操作）"""
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    try:
        encrypted_pwd = encrypt_password(password)
        create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO users (username, password, role, create_time) VALUES (?, ?, ?, ?)", 
                 (username, encrypted_pwd, role, create_time))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False  # 用户名已存在

def verify_user(username, password):
    """验证用户登录"""
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    encrypted_pwd = encrypt_password(password)
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, encrypted_pwd))
    result = c.fetchone()
    conn.close()
    return result is not None

def update_password(username, old_password, new_password):
    """修改密码"""
    if not verify_user(username, old_password):
        return False, "旧密码验证失败"
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    try:
        new_encrypted_pwd = encrypt_password(new_password)
        c.execute("UPDATE users SET password = ? WHERE username = ?", (new_encrypted_pwd, username))
        conn.commit()
        conn.close()
        return True, "密码修改成功"
    except Exception as e:
        conn.close()
        return False, f"修改失败: {str(e)}"

def delete_user(username):
    """删除用户（仅管理员可操作）"""
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    try:
        c.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        conn.close()
        return True, "用户删除成功"
    except Exception as e:
        conn.close()
        return False, f"删除失败: {str(e)}"

def get_all_users():
    """获取所有用户（仅管理员可查看）"""
    conn = sqlite3.connect('user_db.db')
    c = conn.cursor()
    c.execute("SELECT username, role, create_time FROM users ORDER BY create_time DESC")
    users = c.fetchall()
    conn.close()
    return users

# 初始化数据库（首次运行创建表和管理员账号）
init_db()

# ---------------------- 3. 会话状态管理 ----------------------
def init_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.role = None  # 新增：存储用户权限
    if "current_page" not in st.session_state:
        st.session_state.current_page = "login"  # login/main/admin/user_manage/change_pwd
    if "pred_results" not in st.session_state:
        st.session_state.pred_results = None

# 页面切换函数
def go_to_login():
    st.session_state.current_page = "login"
def go_to_main():
    st.session_state.current_page = "main"
def go_to_admin():
    st.session_state.current_page = "admin"
def go_to_user_manage():
    st.session_state.current_page = "user_manage"
def go_to_change_pwd():
    st.session_state.current_page = "change_pwd"

# ---------------------- 4. 自定义CSS ----------------------
st.markdown("""
<style>
* {font-family: "SimHei", "WenQuanYi Micro Hei", "Heiti TC", sans-serif !important;}
body {background-color: #f5f7fa;}
.card {background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); padding: 20px; margin-bottom: 20px;}
.auth-card {max-width: 450px; margin: 30px auto; padding: 30px;}
.section-title {font-size: 18px; font-weight: bold; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px;}
.result-card {border-radius: 8px; padding: 15px; margin: 10px 0; color: white; font-weight: bold;}
.confidence-bar {height: 20px; border-radius: 10px; margin: 5px 0; background: #e0e0e0; overflow: hidden;}
.confidence-fill {height: 100%;}
.stButton>button {background: #3498db !important; color: white !important; border-radius: 6px; padding: 8px 16px; margin: 5px 0;}
.stButton>button:hover {background: #2980b9 !important;}
.stButton>button.secondary {background: #95a5a6 !important;}
.stButton>button.danger {background: #e74c3c !important;}
.link {color: #3498db; text-decoration: underline; cursor: pointer; margin-top: 15px;}
</style>
""", unsafe_allow_html=True)

# ---------------------- 5. 认证页面（仅登录，无公开注册） ----------------------
def login_page():
    st.markdown('<div class="card auth-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">用户登录</h3>', unsafe_allow_html=True)
    
    username = st.text_input("用户名", placeholder="请输入用户名")
    password = st.text_input("密码", type="password", placeholder="请输入密码")
    
    if st.button("登录", use_container_width=True):
        if not username or not password:
            st.error("用户名和密码不能为空")
            return
        if verify_user(username, password):
            # 登录成功，记录权限
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = get_user_role(username)
            go_to_main()
            st.success("登录成功！正在跳转...")
            st.rerun()
        else:
            st.error("用户名或密码错误")
    
    # 仅管理员可见的提示
    st.markdown('<p style="color: #666; text-align: center; margin-top: 20px;">'
                '无账号？请联系管理员创建</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- 6. 管理员功能页（用户管理） ----------------------
def admin_user_manage_page():
    st.title("用户管理（管理员）")
    st.markdown("此处可创建、删除用户，管理系统访问权限")
    
    # 新增用户
    st.markdown('<div class="card"><h3 class="section-title">新增用户</h3>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        new_username = st.text_input("新用户名", placeholder="3-20个字符")
        new_password = st.text_input("初始密码", type="password", placeholder="6-20个字符，含字母和数字")
    with col2:
        new_role = st.selectbox("用户权限", ["user", "admin"])
        confirm_pwd = st.text_input("确认密码", type="password")
    
    def validate_user_input():
        if not new_username or not new_password or not confirm_pwd:
            return False, "所有字段不能为空"
        if len(new_username) < 3 or len(new_username) > 20:
            return False, "用户名长度需在3-20个字符之间"
        if new_password != confirm_pwd:
            return False, "两次输入的密码不一致"
        if not re.search(r'[a-zA-Z]', new_password) or not re.search(r'[0-9]', new_password):
            return False, "密码需同时包含字母和数字"
        return True, ""
    
    if st.button("创建用户", type="primary"):
        valid, msg = validate_user_input()
        if not valid:
            st.error(msg)
        else:
            if add_user(new_username, new_password, new_role):
                st.success(f"用户 {new_username} 创建成功（初始密码：{new_password}）")
            else:
                st.error("用户名已存在")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 查看/删除用户
    st.markdown('<div class="card"><h3 class="section-title">用户列表</h3>', unsafe_allow_html=True)
    users = get_all_users()
    if users:
        user_df = pd.DataFrame(users, columns=["用户名", "权限", "注册时间"])
        st.dataframe(user_df, use_container_width=True)
        
        # 删除用户（禁止删除管理员自己）
        del_username = st.text_input("输入要删除的用户名", placeholder="谨慎操作，不可恢复")
        col_del1, col_del2 = st.columns(2)
        with col_del1:
            if st.button("删除用户", type="primary", key="del_btn"):
                if not del_username:
                    st.error("请输入用户名")
                elif del_username == st.session_state.username:
                    st.error("不能删除当前登录的管理员账号")
                else:
                    success, msg = delete_user(del_username)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        with col_del2:
            if st.button("刷新列表", type="secondary"):
                st.rerun()
    else:
        st.info("暂无用户数据")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 返回主页面
    if st.button("返回系统首页", type="secondary"):
        go_to_main()
        st.rerun()

# ---------------------- 7. 修改密码页面 ----------------------
def change_password_page():
    st.markdown('<div class="card auth-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">修改密码</h3>', unsafe_allow_html=True)
    
    username = st.text_input("用户名", value=st.session_state.username, disabled=True)  # 禁止修改当前用户名
    old_password = st.text_input("旧密码", type="password", placeholder="请输入旧密码")
    new_password = st.text_input("新密码", type="password", placeholder="6-20个字符，含字母和数字")
    confirm_new_pwd = st.text_input("确认新密码", type="password")
    
    def validate_pwd():
        if not old_password or not new_password or not confirm_new_pwd:
            return False, "所有字段不能为空"
        if new_password != confirm_new_pwd:
            return False, "两次输入的新密码不一致"
        if not re.search(r'[a-zA-Z]', new_password) or not re.search(r'[0-9]', new_password):
            return False, "新密码需同时包含字母和数字"
        return True, ""
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("确认修改", use_container_width=True):
            valid, msg = validate_pwd()
            if not valid:
                st.error(msg)
            else:
                success, msg = update_password(username, old_password, new_password)
                if success:
                    st.success(msg)
                    go_to_login()
                    st.session_state.logged_in = False
                    st.rerun()
                else:
                    st.error(msg)
    with col2:
        if st.button("返回首页", use_container_width=True, type="secondary"):
            go_to_main()
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- 8. 主应用功能（预测系统） ----------------------
def main_app():
    # 顶部导航栏（区分管理员和普通用户）
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.title("大豆倒伏级别预测系统")
    with col2:
        if st.button("修改密码", use_container_width=True, type="secondary"):
            go_to_change_pwd()
            st.rerun()
    with col3:
        if st.button("退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = None
            st.session_state.pred_results = None
            go_to_login()
            st.rerun()
    
    # 管理员额外显示"用户管理"入口
    if st.session_state.role == "admin":
        if st.sidebar.button("🔧 用户管理（管理员）", use_container_width=True):
            go_to_user_manage()
            st.rerun()
    
    st.markdown(f"欢迎回来，{st.session_state.username}！请输入大豆特征参数进行预测。")

    # 加载模型和标准化器
    try:
        model = joblib.load('XGBoost.pkl')
        scaler = joblib.load('data_scaler.pkl')
    except FileNotFoundError as e:
        st.error(f"文件未找到: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"加载失败: {str(e)}")
        st.stop()

    # 特征配置
    feature_ranges = {
        '拉力': {"type": "numerical", "min": 0.1, "max": 999999999.0, "default": 18.8, "step": 0.1},
        '株高': {"type": "numerical", "min": 0.1, "max": 999999999.0, "default": 108.0, "step": 1.0},
        '叶柄长': {"type": "numerical", "min": 0.1, "max": 999999999.0, "default": 17.0, "step": 0.1},
        '节数': {"type": "numerical", "min": 1.0, "max": 999999999.0, "default": 17.0, "step": 1.0}
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
        for idx, (feature, props) in enumerate(feature_ranges.items()):
            with cols[idx % 2]:
                st.markdown(f'<div style="display: flex; align-items: center; margin-bottom: 15px;"><div style="width: 100px; padding-right: 10px;">{feature}</div>', unsafe_allow_html=True)
                value = st.number_input(
                    feature,
                    min_value=float(props["min"]),
                    max_value=float(props["max"]),
                    value=float(props["default"]),
                    step=props["step"],
                    label_visibility="collapsed"
                )
                feature_values[feature] = value
                st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 预测与可视化
    if st.button("预测倒伏级别", type="primary", use_container_width=True):
        try:
            株高拉力比 = feature_values['株高'] / feature_values['拉力']
            叶柄节数比 = feature_values['叶柄长'] / feature_values['节数']
        except Exception as e:
            st.error(f"特征计算错误: {str(e)}")
            st.stop()
        
        try:
            model_input = [feature_values['拉力'], feature_values['株高'], feature_values['叶柄长'],
                          feature_values['节数'], 株高拉力比, 叶柄节数比]
            input_scaled = scaler.transform([model_input])
            input_data = pd.DataFrame(input_scaled, columns=model_feature_names)
        except Exception as e:
            st.error(f"数据标准化错误: {str(e)}")
            st.stop()
        
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
            st.warning(f"SHAP值计算失败: {str(e)}")
            single_shap = None
            base_value = None

        st.session_state.pred_results = {
            "pred_class": pred_class, "pred_proba": pred_proba, "confidence": confidence,
            "single_shap": single_shap, "feature_names": model_feature_names,
            "english_feature_names": english_feature_names, "feature_values": model_input,
            "base_value": base_value, "input_features": feature_values,
            "derived_features": {"株高拉力比": 株高拉力比, "叶柄节数比": 叶柄节数比}
        }

    # 显示预测结果
    if st.session_state.pred_results is not None:
        res = st.session_state.pred_results
        pred_class = int(res["pred_class"])
        
        st.markdown("### 输入特征与计算特征")
        with st.expander("查看详细特征值", expanded=True):
            input_df = pd.DataFrame(list(res["input_features"].items()), columns=["特征", "值"])
            st.dataframe(input_df, use_container_width=True)
            derived_df = pd.DataFrame(list(res["derived_features"].items()), columns=["特征", "值"])
            derived_df["值"] = derived_df["值"].apply(lambda x: f"{x:.4f}")
            st.dataframe(derived_df, use_container_width=True)

        st.markdown("### 倒伏级别预测结果")
        level_info = lodging_levels.get(pred_class, {"name": f"级别{pred_class}", "description": "无说明"})
        level_colors = {0: "#2ecc71", 1: "#3498db", 2: "#f39c12", 3: "#e74c3c"}
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
        
        st.markdown("#### 各级别概率分布")
        prob_data = []
        for i, prob in enumerate(res["pred_proba"]):
            level_name = lodging_levels.get(i, {"name": f"级别{i}"})["name"]
            prob_data.append({"倒伏级别": f"{i} ({level_name})", "概率": f"{prob*100:.2f}%"})
        st.dataframe(pd.DataFrame(prob_data), use_container_width=True)

        if res["single_shap"] is not None:
            st.markdown("### SHAP特征贡献图 (Feature Contribution)")
            st.markdown("蓝色 = 降低倒伏风险，红色 = 增加倒伏风险，长度 = 贡献程度")
            
            shap_exp = shap.Explanation(
                values=res['single_shap'], base_values=res['base_value'],
                data=res['feature_values'], feature_names=res['english_feature_names']
            )

            plt.figure(figsize=(12, 8))
            shap.plots.waterfall(shap_exp, max_display=10, show=False)
            for ax in plt.gcf().axes:
                ax.set_title(ax.get_title(), fontfamily="Times New Roman")
                ax.set_xlabel(ax.get_xlabel(), fontfamily="Times New Roman")
                ax.set_ylabel(ax.get_ylabel(), fontfamily="Times New Roman")
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontfamily("Times New Roman")
            plt.tight_layout()
            st.pyplot(plt.gcf())

            if st.checkbox("显示所有特征的SHAP值", key="show_shap"):
                shap_df = pd.DataFrame({
                    "特征 (Feature)": [f"{cn} ({en})" for cn, en in zip(res['feature_names'], res['english_feature_names'])],
                    "数值 (Value)": [round(v, 4) for v in res['feature_values']],
                    "SHAP值 (贡献度)": res['single_shap'].round(4)
                })
                st.dataframe(shap_df, use_container_width=True)

# ---------------------- 9. 程序入口 ----------------------
if __name__ == "__main__":
    init_session_state()
    
    # 根据页面状态显示内容
    if st.session_state.current_page == "login":
        login_page()
    elif st.session_state.current_page == "main":
        main_app()
    elif st.session_state.current_page == "user_manage" and st.session_state.role == "admin":
        admin_user_manage_page()
    elif st.session_state.current_page == "change_pwd":
        change_password_page()
    else:
        # 非法访问跳转登录
        go_to_login()
        st.rerun()
