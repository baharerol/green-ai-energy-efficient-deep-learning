import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import numpy as np

#Sayfa Ayarları
st.set_page_config(
    page_title="Green AI Dashboard",
    layout="wide"
)

#Custom CSS (Dark Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #0f1117;
        color: #e0e0e0;
    }
    
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #f0f0f0;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-top: 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        text-align: center;
    }
    .metric-card.green {
        border-color: #22c55e;
        background: linear-gradient(135deg, #0f2a1a 0%, #14532d 100%);
    }
    .metric-card.blue {
        border-color: #3b82f6;
        background: linear-gradient(135deg, #0f1a2e 0%, #1e3a5f 100%);
    }
    .metric-card.orange {
        border-color: #f97316;
        background: linear-gradient(135deg, #2a1a0f 0%, #5f3a1e 100%);
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .metric-card .label {
        font-size: 0.8rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .detail {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-top: 0.2rem;
    }
    
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #f0f0f0;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #334155;
    }
    
    .insight-box {
        background: #0d2818;
        border-left: 4px solid #22c55e;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #d1d5db;
    }
    .insight-box.warning {
        background: #2a1f0a;
        border-left-color: #f59e0b;
    }
    
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #334155;
        margin-top: 3rem;
    }
    
    .co2-box {
        background: #1e293b;
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .co2-box .big {
        font-size: 2rem;
        font-weight: 700;
        color: #4ade80;
    }
    .co2-box .sub {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 0.3rem;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 20px; }
    .stSelectbox label, .stMultiSelect label { color: #e0e0e0 !important; }
    div[data-testid="stSidebar"] { background-color: #1a1d29; }
</style>
""", unsafe_allow_html=True)

#Veri
df = pd.read_csv("notebooks/results/all_results.csv")
df_mob = df[df['model'] == 'MobileNetV2'].copy()
df_dis = df[df['model'] == 'DistilBERT'].copy()
df_lstm = df[df['model'] == 'LSTM'].copy()
LSTM_COLOR = '#10B981'

# Plotly dark tema renkleri
MOB_COLOR = '#60A5FA'
DIS_COLOR = '#FB923C'
GREEN = '#4ADE80'
RED = '#F87171'
GRAY = '#6B7280'
BG_DARK = '#0f1117'
CARD_BG = '#1e293b'

DARK_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", size=12, color='#e0e0e0'),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=20, r=20, t=50, b=20),
    height=400,
)

#Header
st.markdown("""
<div class="main-header">
    <h1>Green AI: Enerji Verimli Derin Öğrenme</h1>
    <p>MobileNetV2 + CIFAR-10 | DistilBERT + SST-2 | LSTM + NASA's Sunspot Time Series | Optimizasyon Tekniklerinin Karşılaştırmalı Analizi</p>
</div>
""", unsafe_allow_html=True)

#Özet Metrikler
best_acc_row = df.loc[df['accuracy'].idxmax()]
smallest_row = df.loc[df['model_size_mb'].idxmin()]
df_energy_valid = df[df['energy_kwh'] > 0.001]
lowest_energy_row = df_energy_valid.loc[df_energy_valid['energy_kwh'].idxmin()] if len(df_energy_valid) > 0 else None

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card green">
        <div class="label">En Yüksek Doğruluk</div>
        <div class="value">{best_acc_row['accuracy']:.2%}</div>
        <div class="detail">{best_acc_row['experiment']}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card blue">
        <div class="label">En Küçük Model</div>
        <div class="value">{smallest_row['model_size_mb']:.2f} MB</div>
        <div class="detail">{smallest_row['experiment']}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    if lowest_energy_row is not None:
        st.markdown(f"""
        <div class="metric-card orange">
            <div class="label">En Az Enerji</div>
            <div class="value">{lowest_energy_row['energy_kwh']*1000:.2f} Wh</div>
            <div class="detail">{lowest_energy_row['experiment']}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

#Tabs
tab1, tab2, tab3, tab4, tab_lstm, tab5, tab6 = st.tabs([
    "Doğruluk", "Model Boyutu", "Enerji", "Çıkarım Hızı", "LSTM", "Deney Karşılaştırıcı", "Veri Tablosu"
])

#TAB 1: Doğruluk 
with tab1:
    st.markdown('<div class="section-header">Doğruluk Karşılaştırması</div>', unsafe_allow_html=True)
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        dm_sorted = df_mob.sort_values('accuracy', ascending=True)
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            y=dm_sorted['experiment'], x=dm_sorted['accuracy'],
            orientation='h', marker_color=MOB_COLOR,
            text=[f'{v:.4f}' for v in dm_sorted['accuracy']],
            textposition='outside', textfont=dict(size=11, color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>Doğruluk: %{x:.4f}<extra></extra>'
        ))
        baseline_acc = dm_sorted[dm_sorted['experiment'].str.contains('Unfreezed')]['accuracy']
        if len(baseline_acc) > 0:
            fig1.add_vline(x=baseline_acc.values[0], line_dash="dash", 
                          line_color=GRAY, annotation_text="Best Baseline",
                          annotation_font_color=GRAY)
        fig1.update_layout(**DARK_LAYOUT, title=dict(text="MobileNetV2 + CIFAR-10", font_color='#e0e0e0'),
                          xaxis=dict(range=[0, 1.05], title="Doğruluk", gridcolor='#1e293b'))
        st.plotly_chart(fig1, use_container_width=True)
    
    with col_r:
        dd_sorted = df_dis.sort_values('accuracy', ascending=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=dd_sorted['experiment'], x=dd_sorted['accuracy'],
            orientation='h', marker_color=DIS_COLOR,
            text=[f'{v:.4f}' for v in dd_sorted['accuracy']],
            textposition='outside', textfont=dict(size=11, color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>Doğruluk: %{x:.4f}<extra></extra>'
        ))
        baseline_acc_d = dd_sorted[dd_sorted['experiment'].str.contains('Baseline')].iloc[0]['accuracy']
        fig2.add_vline(x=baseline_acc_d, line_dash="dash", 
                      line_color=GRAY, annotation_text="Baseline",
                      annotation_font_color=GRAY)
        fig2.update_layout(**DARK_LAYOUT, title=dict(text="DistilBERT + SST-2", font_color='#e0e0e0'),
                          xaxis=dict(range=[0, 1.05], title="Doğruluk", gridcolor='#1e293b'))
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>Bulgu:</strong> Her iki modelde de early stopping ve structured pruning, baseline doğruluğunu 
        koruyarak veya artırarak optimizasyon sağlamıştır. Quantization her iki modelde de doğruluk kaybına yol açmıştır.
    </div>
    """, unsafe_allow_html=True)

# TAB 2: Model Boyutu

with tab2:
    st.markdown('<div class="section-header">Model Boyutu Karşılaştırması</div>', unsafe_allow_html=True)
    
    col_l, col_m, col_r = st.columns(3)
    
    with col_l:
        dm_s = df_mob.sort_values('model_size_mb', ascending=True)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            y=dm_s['experiment'], x=dm_s['model_size_mb'],
            orientation='h', marker_color=MOB_COLOR,
            text=[f'{v:.2f} MB' for v in dm_s['model_size_mb']],
            textposition='outside', textfont=dict(size=11, color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>Boyut: %{x:.2f} MB<extra></extra>'
        ))
        fig3.update_layout(**DARK_LAYOUT, title=dict(text="MobileNetV2", font_color='#e0e0e0'),
                          xaxis=dict(title="Boyut (MB)", gridcolor='#1e293b'))
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_m:
        dd_s = df_dis.sort_values('model_size_mb', ascending=True)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            y=dd_s['experiment'], x=dd_s['model_size_mb'],
            orientation='h', marker_color=DIS_COLOR,
            text=[f'{v:.2f} MB' for v in dd_s['model_size_mb']],
            textposition='outside', textfont=dict(size=11, color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>Boyut: %{x:.2f} MB<extra></extra>'
        ))
        fig4.update_layout(**DARK_LAYOUT, title=dict(text="DistilBERT", font_color='#e0e0e0'),
                          xaxis=dict(title="Boyut (MB)", gridcolor='#1e293b'))
        st.plotly_chart(fig4, use_container_width=True)
    
    with col_r:
        dl_s = df_lstm.sort_values('model_size_mb', ascending=True)
        fig4b = go.Figure()
        fig4b.add_trace(go.Bar(
            y=dl_s['experiment'], x=dl_s['model_size_mb'],
            orientation='h', marker_color=LSTM_COLOR,
            text=[f'{v:.4f} MB' for v in dl_s['model_size_mb']],
            textposition='outside', textfont=dict(size=11, color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>Boyut: %{x:.4f} MB<extra></extra>'
        ))
        fig4b.update_layout(**DARK_LAYOUT, title=dict(text="LSTM", font_color='#e0e0e0'),
                           xaxis=dict(title="Boyut (MB)", gridcolor='#1e293b'))
        st.plotly_chart(fig4b, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>Bulgu:</strong> Quantization (INT8) her üç modelde de en büyük boyut azalmasını sağlamıştır — 
        MobileNetV2'de %72, DistilBERT'te %48, LSTM'de %74. LSTM'de hidden size reduction da 
        benzer boyut kazancı sağlamıştır (%75) ancak farklı mekanizmayla: parametre sayısını azaltarak.
    </div>
    """, unsafe_allow_html=True)
    

#TAB 3: Enerji
with tab3:
    st.markdown('<div class="section-header">Eğitim Enerji Tüketimi</div>', unsafe_allow_html=True)
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        dm_e = df_mob[~df_mob['experiment'].str.contains('Quantization', na=False)]
        dm_e = dm_e[dm_e['energy_kwh'] > 0.001].sort_values('energy_kwh', ascending=True)
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            y=dm_e['experiment'], x=dm_e['energy_kwh'] * 1000,
            orientation='h', marker_color=MOB_COLOR,
            text=[f'{v*1000:.2f} Wh' for v in dm_e['energy_kwh']],
            textposition='outside', textfont=dict(size=11, color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>Enerji: %{x:.2f} Wh<extra></extra>'
        ))
        fig5.update_layout(**DARK_LAYOUT, title=dict(text="MobileNetV2", font_color='#e0e0e0'),
                          xaxis=dict(title="Enerji (Wh)", gridcolor='#1e293b'))
        st.plotly_chart(fig5, use_container_width=True)
    
    with col_r:
        dd_e = df_dis[df_dis['energy_kwh'] > 0.001].sort_values('energy_kwh', ascending=True)
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            y=dd_e['experiment'], x=dd_e['energy_kwh'] * 1000,
            orientation='h', marker_color=DIS_COLOR,
            text=[f'{v*1000:.2f} Wh' for v in dd_e['energy_kwh']],
            textposition='outside', textfont=dict(size=11, color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>Enerji: %{x:.2f} Wh<extra></extra>'
        ))
        fig6.update_layout(**DARK_LAYOUT, title=dict(text="DistilBERT (yalnızca eğitim içeren deneyler)", font_color='#e0e0e0'),
                          xaxis=dict(title="Enerji (Wh)", gridcolor='#1e293b'))
        st.plotly_chart(fig6, use_container_width=True)
    
    # CO2 somutlaştırma
    total_energy = df_energy_valid['energy_kwh'].sum()
    total_co2 = df['co2_kg'].sum() if 'co2_kg' in df.columns and df['co2_kg'].notna().any() else total_energy * 0.4647
    phone_charges = total_energy * 1000 / 15
    
    st.markdown(f"""
    <div class="co2-box">
        <div class="big">{total_energy*1000:.1f} Wh</div>
        <div class="sub">Tüm deneylerin toplam enerji tüketimi</div>
        <div class="sub" style="margin-top: 0.8rem;">
            ≈ {phone_charges:.1f} telefon şarjı | ≈ {total_co2*1000:.1f} g CO₂
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box warning">
        <strong>Not:</strong> Quantization enerji değerleri grafiğe dahil edilmemiştir — bu değerler eğitim değil, 
        model dönüşüm sürecini yansıtmaktadır. Early stopping enerji tüketimi baseline'dan yüksek görünmektedir 
        çünkü farklı epoch sayılarıyla çalışmıştır (4 vs 3 epoch).
    </div>
    """, unsafe_allow_html=True)

#TAB 4: Çıkarım Hızı


with tab4:
    st.markdown('<div class="section-header">Çıkarım Süresi Karşılaştırması</div>', unsafe_allow_html=True)
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        dd_inf = df_dis[df_dis['inference_time_s'].notna()].sort_values('inference_time_s', ascending=True)
        colors = [GREEN if val == dd_inf['inference_time_s'].min() 
                  else (RED if val == dd_inf['inference_time_s'].max() else DIS_COLOR) 
                  for val in dd_inf['inference_time_s']]
        
        fig7 = go.Figure()
        fig7.add_trace(go.Bar(
            y=dd_inf['experiment'], x=dd_inf['inference_time_s'],
            orientation='h', marker_color=colors,
            text=[f'{v:.4f}s' for v in dd_inf['inference_time_s']],
            textposition='outside', textfont=dict(size=11, color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>Çıkarım: %{x:.4f}s<extra></extra>'
        ))
        baseline_inf = dd_inf[dd_inf['experiment'].str.contains('Baseline')]['inference_time_s']
        if len(baseline_inf) > 0:
            fig7.add_vline(x=baseline_inf.values[0], line_dash="dash", 
                          line_color=GRAY, 
                          annotation_text=f"Baseline ({baseline_inf.values[0]:.4f}s)",
                          annotation_font_color=GRAY)
        fig7.update_layout(**DARK_LAYOUT, title=dict(text="DistilBERT — Çıkarım Süresi", font_color='#e0e0e0'),
                          xaxis=dict(title="Süre (s)", gridcolor='#1e293b'))
        st.plotly_chart(fig7, use_container_width=True)
    
    with col_r:
        lstm_inf = df_lstm[df_lstm['inference_time_s'].notna()].sort_values('inference_time_s', ascending=True)
        colors_lstm = [GREEN if val == lstm_inf['inference_time_s'].min()
                       else (RED if val == lstm_inf['inference_time_s'].max() else LSTM_COLOR)
                       for val in lstm_inf['inference_time_s']]
        fig7b = go.Figure()
        fig7b.add_trace(go.Bar(
            y=lstm_inf['experiment'], x=lstm_inf['inference_time_s'] * 1000,
            orientation='h', marker_color=colors_lstm,
            text=[f'{v*1000:.2f} ms' for v in lstm_inf['inference_time_s']],
            textposition='outside', textfont=dict(size=11, color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>Çıkarım: %{x:.2f} ms<extra></extra>'
        ))
        baseline_linf = lstm_inf[lstm_inf['experiment'].str.contains('Baseline')]['inference_time_s']
        if len(baseline_linf) > 0:
            fig7b.add_vline(x=baseline_linf.values[0] * 1000, line_dash="dash",
                           line_color=GRAY,
                           annotation_text=f"Baseline ({baseline_linf.values[0]*1000:.2f} ms)",
                           annotation_font_color=GRAY)
        fig7b.update_layout(**DARK_LAYOUT, title=dict(text="LSTM — Çıkarım Süresi", font_color='#e0e0e0'),
                           xaxis=dict(title="Süre (ms)", gridcolor='#1e293b'))
        st.plotly_chart(fig7b, use_container_width=True)
    
    # Early stopping grafiği ayrı satırda
    st.markdown('<div class="section-header">DistilBERT — Early Stopping Eğitim Seyri</div>', unsafe_allow_html=True)
    
    epochs = [1, 2, 3, 4]
    train_loss = [0.3865, 0.1932, 0.1072, 0.0513]
    val_loss = [0.3166, 0.3076, 0.3468, 0.4651]
    
    fig8 = go.Figure()
    fig8.add_trace(go.Scatter(
        x=epochs, y=train_loss, mode='lines+markers',
        name='Eğitim Kaybı', line=dict(color=MOB_COLOR, width=3),
        marker=dict(size=10)
    ))
    fig8.add_trace(go.Scatter(
        x=epochs, y=val_loss, mode='lines+markers',
        name='Doğrulama Kaybı', line=dict(color=DIS_COLOR, width=3),
        marker=dict(size=10, symbol='square')
    ))
    fig8.add_vline(x=2, line_dash="dash", line_color=GREEN,
                  annotation_text="En iyi epoch", annotation_font_color=GREEN)
    fig8.add_vline(x=4, line_dash="dash", line_color=RED,
                  annotation_text="Durdurulma", annotation_font_color=RED)
    fig8.add_vrect(x0=2, x1=4, fillcolor="red", opacity=0.08)
    fig8.update_layout(**DARK_LAYOUT, title=dict(text="Early Stopping — Eğitim Seyri", font_color='#e0e0e0'),
                      xaxis=dict(title="Epoch", dtick=1, gridcolor='#1e293b'),
                      yaxis=dict(title="Kayıp (Loss)", gridcolor='#1e293b'),
                      legend=dict(x=0.55, y=0.95, font_color='#e0e0e0'),
                      )
    st.plotly_chart(fig8, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>Bulgu:</strong> Dynamic quantization her iki modelde de çıkarım süresini artırmıştır: 
        DistilBERT'te %40, LSTM'de %160. Buna karşılık yapısal küçültme (structured pruning / hidden size reduction) 
        her iki modelde de gerçek hızlanma sağlamıştır: DistilBERT'te %10, LSTM'de %61.
    </div>
    """, unsafe_allow_html=True)
    
    
#TAB LSTM 
with tab_lstm:
    st.markdown('<div class="section-header">LSTM + Güneş Lekesi Zaman Serisi</div>', unsafe_allow_html=True)
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        # MAE karşılaştırması
        lstm_sorted = df_lstm.sort_values('mae', ascending=True)
        fig_mae = go.Figure()
        fig_mae.add_trace(go.Bar(
            y=lstm_sorted['experiment'], x=lstm_sorted['mae'],
            orientation='h', marker_color=LSTM_COLOR,
            text=[f'{v:.2f}' for v in lstm_sorted['mae']],
            textposition='outside', textfont=dict(size=11, color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>MAE: %{x:.2f}<extra></extra>'
        ))
        baseline_mae = df_lstm[df_lstm['experiment'].str.contains('Baseline')]['mae'].values[0]
        fig_mae.add_vline(x=baseline_mae, line_dash="dash", line_color=GRAY,
                         annotation_text=f"Baseline ({baseline_mae:.2f})",
                         annotation_font_color=GRAY)
        fig_mae.update_layout(**DARK_LAYOUT, title=dict(text="MAE Karşılaştırması", font_color='#e0e0e0'),
                             xaxis=dict(title="MAE (Güneş Lekesi Sayısı)", gridcolor='#1e293b',
                                       range=[35, 45]))
        st.plotly_chart(fig_mae, use_container_width=True)
    
    with col_r:
        # RMSE karşılaştırması
        lstm_sorted_r = df_lstm.sort_values('rmse', ascending=True)
        fig_rmse = go.Figure()
        fig_rmse.add_trace(go.Bar(
            y=lstm_sorted_r['experiment'], x=lstm_sorted_r['rmse'],
            orientation='h', marker_color=LSTM_COLOR,
            text=[f'{v:.2f}' for v in lstm_sorted_r['rmse']],
            textposition='outside', textfont=dict(size=11, color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>RMSE: %{x:.2f}<extra></extra>'
        ))
        baseline_rmse = df_lstm[df_lstm['experiment'].str.contains('Baseline')]['rmse'].values[0]
        fig_rmse.add_vline(x=baseline_rmse, line_dash="dash", line_color=GRAY,
                          annotation_text=f"Baseline ({baseline_rmse:.2f})",
                          annotation_font_color=GRAY)
        fig_rmse.update_layout(**DARK_LAYOUT, title=dict(text="RMSE Karşılaştırması", font_color='#e0e0e0'),
                              xaxis=dict(title="RMSE (Güneş Lekesi Sayısı)", gridcolor='#1e293b',
                                        range=[45, 55]))
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    col_l2, col_r2 = st.columns(2)
    
    with col_l2:
        # Çıkarım süresi
        lstm_inf = df_lstm.sort_values('inference_time_s', ascending=True)
        colors_lstm = [GREEN if val == lstm_inf['inference_time_s'].min()
                       else (RED if val == lstm_inf['inference_time_s'].max() else LSTM_COLOR)
                       for val in lstm_inf['inference_time_s']]
        fig_linf = go.Figure()
        fig_linf.add_trace(go.Bar(
            y=lstm_inf['experiment'], x=lstm_inf['inference_time_s'] * 1000,
            orientation='h', marker_color=colors_lstm,
            text=[f'{v*1000:.2f} ms' for v in lstm_inf['inference_time_s']],
            textposition='outside', textfont=dict(size=11, color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>Çıkarım: %{x:.2f} ms<extra></extra>'
        ))
        fig_linf.update_layout(**DARK_LAYOUT, title=dict(text="Çıkarım Süresi", font_color='#e0e0e0'),
                              xaxis=dict(title="Süre (ms)", gridcolor='#1e293b'))
        st.plotly_chart(fig_linf, use_container_width=True)
    
    with col_r2:
        # Model boyutu
        lstm_size = df_lstm.sort_values('model_size_mb', ascending=True)
        fig_lsize = go.Figure()
        fig_lsize.add_trace(go.Bar(
            y=lstm_size['experiment'], x=lstm_size['model_size_mb'],
            orientation='h', marker_color=LSTM_COLOR,
            text=[f'{v:.4f} MB' for v in lstm_size['model_size_mb']],
            textposition='outside', textfont=dict(size=11, color='#e0e0e0'),
            hovertemplate='<b>%{y}</b><br>Boyut: %{x:.4f} MB<extra></extra>'
        ))
        fig_lsize.update_layout(**DARK_LAYOUT, title=dict(text="Model Boyutu", font_color='#e0e0e0'),
                               xaxis=dict(title="Boyut (MB)", gridcolor='#1e293b'))
        st.plotly_chart(fig_lsize, use_container_width=True)
    
    st.markdown("""
    <div class="insight-box">
        <strong>Bulgu:</strong> Küçük model (hidden=128) en dengeli ödünleşimi sağlamıştır: %74.8 boyut azalması, 
        %60.7 çıkarım hızlanması ve yalnızca %6.8 MAE artışı. Dynamic quantization dosya boyutunu benzer oranda 
        düşürmüş ancak çıkarım süresini %160 artırmıştır — CPU'da tutarlı olarak gözlemlenen cross-model pattern.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box warning">
        <strong>Not:</strong> LSTM enerji tüketimi (0.01–0.04 Wh) CodeCarbon'un ölçüm hassasiyetinin alt sınırındadır. 
        Bu nedenle LSTM deneyleri model boyutu ve çıkarım hızı odaklı değerlendirilmiştir.
    </div>
    """, unsafe_allow_html=True)

#TAB 5: Deney Karşılaştırıcı
with tab5:
    st.markdown('<div class="section-header">Deney Karşılaştırıcı</div>', unsafe_allow_html=True)
    
    model_choice = st.radio("Model", ["MobileNetV2", "DistilBERT", "LSTM"], horizontal=True)
    df_filtered = df[df['model'] == model_choice]
    filtered_experiments = df_filtered['experiment'].tolist()
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        exp1 = st.selectbox("1. Deney", filtered_experiments, index=0, key="exp1")
    with col_sel2:
        default_idx = min(1, len(filtered_experiments) - 1)
        exp2 = st.selectbox("2. Deney", filtered_experiments, index=default_idx, key="exp2")
    
    row1 = df_filtered[df_filtered['experiment'] == exp1].iloc[0]
    row2 = df_filtered[df_filtered['experiment'] == exp2].iloc[0]
    
    def compare_val(v1, v2, higher_is_better=True):
        if pd.isna(v1) or pd.isna(v2):
            return "neutral", "\u2014", "\u2014"
        diff = v2 - v1
        pct = (diff / v1 * 100) if v1 != 0 else 0
        if higher_is_better:
            cls = "better" if diff > 0 else ("worse" if diff < 0 else "neutral")
        else:
            cls = "better" if diff < 0 else ("worse" if diff > 0 else "neutral")
        sign = "+" if diff > 0 else ""
        return cls, f"{sign}{pct:.1f}%", f"{sign}{diff:.4f}"
    
    if model_choice == "LSTM":
        metrics = [
            ("MAE", 'mae', False, ""),
            ("RMSE", 'rmse', False, ""),
            ("Model Boyutu", 'model_size_mb', False, " MB"),
            ("Enerji", 'energy_kwh', False, " kWh"),
            ("Çıkarım Süresi", 'inference_time_s', False, " s"),
        ]
    else:
        metrics = [
            ("Doğruluk", 'accuracy', True, ""),
            ("F1 Skoru", 'f1_score', True, ""),
            ("Model Boyutu", 'model_size_mb', False, " MB"),
            ("Enerji", 'energy_kwh', False, " kWh"),
        ]
        if 'inference_time_s' in df.columns:
            metrics.append(("Çıkarım Süresi", 'inference_time_s', False, " s"))
    
    comparison_html = f"""
    <div style="display: grid; grid-template-columns: 1fr 2fr 1fr 2fr 1fr; gap: 0; 
                background: {CARD_BG}; border-radius: 12px; border: 1px solid #475569; overflow: hidden;">
        <div style="padding: 1rem; text-align: center; font-weight: 600; background: #334155; color: #9ca3af;">Metrik</div>
        <div style="padding: 1rem; text-align: center; font-weight: 600; background: #334155; color: {MOB_COLOR};">{exp1}<br><span style="font-size:0.75rem; color:#9ca3af;">{row1['model']}</span></div>
        <div style="padding: 1rem; text-align: center; font-weight: 600; background: #334155; color: #9ca3af;">Fark</div>
        <div style="padding: 1rem; text-align: center; font-weight: 600; background: #334155; color: {DIS_COLOR};">{exp2}<br><span style="font-size:0.75rem; color:#9ca3af;">{row2['model']}</span></div>
        <div style="padding: 1rem; text-align: center; font-weight: 600; background: #334155; color: #9ca3af;">Değişim</div>
    """
    
    for label, col, higher_better, unit in metrics:
        v1 = row1.get(col, None)
        v2 = row2.get(col, None)
        cls, pct_str, diff_str = compare_val(v1, v2, higher_better)
        
        v1_str = f"{v1:.4f}{unit}" if pd.notna(v1) else "\u2014"
        v2_str = f"{v2:.4f}{unit}" if pd.notna(v2) else "\u2014"
        
        color_map = {"better": "#4ade80", "worse": "#f87171", "neutral": "#9ca3af"}
        
        comparison_html += f"""
        <div style="padding: 0.8rem 1rem; border-top: 1px solid #334155; color: #d1d5db;">{label}</div>
        <div style="padding: 0.8rem 1rem; border-top: 1px solid #334155; text-align: center; color: #e0e0e0; font-family: monospace;">{v1_str}</div>
        <div style="padding: 0.8rem 1rem; border-top: 1px solid #334155; text-align: center; color: {color_map[cls]}; font-weight: 600;">{pct_str}</div>
        <div style="padding: 0.8rem 1rem; border-top: 1px solid #334155; text-align: center; color: #e0e0e0; font-family: monospace;">{v2_str}</div>
        <div style="padding: 0.8rem 1rem; border-top: 1px solid #334155; text-align: center; color: {color_map[cls]};">{diff_str}</div>
        """
    
    comparison_html += "</div>"

    
    comparison_html += "</div>"
    components.html(f"""
    <div style="font-family: Inter, sans-serif; background: transparent;">
    {comparison_html}
    </div>
    """, height=450, scrolling=False)
    

    # Radar Chart
    st.markdown("<br>", unsafe_allow_html=True)
    
    def normalize_for_radar(val, col_name, invert=False):
        col_vals = df[col_name].dropna()
        if len(col_vals) == 0:
            return 0
        min_v, max_v = col_vals.min(), col_vals.max()
        if max_v == min_v:
            return 0.5
        norm = (val - min_v) / (max_v - min_v)
        return 1 - norm if invert else norm
    
    if model_choice == "LSTM":
        categories = ['Tahmin\nDoğruluğu', 'RMSE\nDoğruluğu', 'Model\nKompaktlığı', 'Çıkarım\nHızı']
        radar_cols = [('mae', True), ('rmse', True), ('model_size_mb', True), ('inference_time_s', True)]
    else:
        categories = ['Doğruluk', 'F1 Skoru', 'Model\nKompaktlığı', 'Enerji\nVerimliliği']
        radar_cols = [('accuracy', False), ('f1_score', False), ('model_size_mb', True), ('energy_kwh', True)]
    
    vals1, vals2 = [], []
    for col, invert in radar_cols:
        v1 = row1.get(col, None)
        v2 = row2.get(col, None)
        vals1.append(normalize_for_radar(v1, col, invert) if pd.notna(v1) and v1 > 0.001 else 0)
        vals2.append(normalize_for_radar(v2, col, invert) if pd.notna(v2) and v2 > 0.001 else 0)
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals1 + [vals1[0]], theta=categories + [categories[0]],
        fill='toself', name=exp1,
        line_color=MOB_COLOR, fillcolor='rgba(96,165,250,0.15)'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=vals2 + [vals2[0]], theta=categories + [categories[0]],
        fill='toself', name=exp2,
        line_color=DIS_COLOR, fillcolor='rgba(251,146,60,0.15)'
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#334155', 
                           tickfont_color='#6b7280'),
            angularaxis=dict(gridcolor='#334155', tickfont=dict(color='#d1d5db', size=11))
        ),
        font=dict(family="Inter, sans-serif", color='#e0e0e0'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font_color='#e0e0e0'),
        height=450,
        title=dict(text="Çok Boyutlu Karşılaştırma", font=dict(color='#e0e0e0', size=14))
    )
    st.plotly_chart(fig_radar, use_container_width=True)

#TAB 6: Veri Tablosu
with tab6:
    st.markdown('<div class="section-header">Tüm Deney Sonuçları</div>', unsafe_allow_html=True)
    
    df_display = df[['model', 'experiment', 'accuracy', 'f1_score', 
                     'model_size_mb', 'energy_kwh', 'mae', 'rmse']].copy()
    df_display['energy_display'] = df_display['energy_kwh'].apply(
        lambda x: f"{x*1000:.2f} Wh" if pd.notna(x) and x > 0.001 else "\u2014"
    )
    df_display = df_display.drop(columns=['energy_kwh'])
    df_display.columns = ['Model', 'Deney', 'Doğruluk', 'F1 Skoru', 
                           'Boyut (MB)', 'MAE', 'RMSE', 'Enerji']
    
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Doğruluk": st.column_config.ProgressColumn(
                format="%.4f", min_value=0, max_value=1
            ),
            "F1 Skoru": st.column_config.ProgressColumn(
                format="%.4f", min_value=0, max_value=1
            ),
        }
    )

# Footer
st.markdown("""
<div class="footer">
    TÜBİTAK 2209-A | Green AI — Enerji Verimli Derin Öğrenme Optimizasyonları<br>
    Ortam: Windows CPU-only | CodeCarbon OfflineEmissionsTracker | PyTorch + TensorFlow | 3 Model, 13 Deney
</div>
""", unsafe_allow_html=True)