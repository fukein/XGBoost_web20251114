# ---------------------- 1. 基础配置（增强中文显示） ----------------------
import matplotlib
# 优先使用支持中文的字体，确保覆盖系统默认
matplotlib.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ---------------------- 5. 预测与SHAP可视化（修改SHAP绘图部分） ----------------------
if res["single_shap"] is not None and res["base_value"] is not None:
    st.markdown("### SHAP特征贡献图")
    st.markdown("蓝色 = 降低倒伏风险，红色 = 增加倒伏风险，长度 = 贡献程度（前10个特征）")
    
    # 单独为SHAP设置字体（关键：覆盖SHAP内部的字体配置）
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
    
    shap_exp = shap.Explanation(
        values=res['single_shap'],
        base_values=res['base_value'],
        data=res['feature_values'],
        feature_names=res['feature_names']
    )

    plt.figure(figsize=(12, 8))
    # 绘制SHAP瀑布图时强制指定字体
    shap.plots.waterfall(shap_exp, max_display=10, show=False)
    
    # 额外设置坐标轴字体（防止SHAP内部覆盖）
    for ax in plt.gcf().axes:
        ax.set_title(ax.get_title(), fontproperties="SimHei")
        ax.set_xlabel(ax.get_xlabel(), fontproperties="SimHei")
        ax.set_ylabel(ax.get_ylabel(), fontproperties="SimHei")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties("SimHei")
    
    plt.tight_layout()
    st.pyplot(plt.gcf())
