import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy import stats
import ipywidgets as widgets
import kagglehub
from IPython.display import display, clear_output
import time
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ========================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ========================================================
central_asia = ['KAZAKHSTAN', 'KYRGYZSTAN', 'TAJIKISTAN', 'TURKMENISTAN', 'UZBEKISTAN']
year_cols = [str(y) for y in range(1995, 2023)]
RESULTS_DIR = "results"
COMPARISON_DIR = "comparison"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)

df = None
df_melted = None
df_cluster = None
future_predictions = None


# ========================================================
# 1. def load_data_safely()
# ========================================================
def load_data_safely():
    """Безопасная загрузка + полный анализ структуры датасета UNWTO"""
    global df, df_melted
    try:
        print("Загрузка данных с Kaggle (UNWTO Tourism Data)...")
        # path = kagglehub.dataset_download("tronheim/unwto-tourism-data-structured-for-analysis")
        csv_path = 'structured_UNWTO_tourism_data.csv'
        df = pd.read_csv(csv_path)

        # Приводим 2022 к числу
        df['2022'] = pd.to_numeric(df['2022'], errors='coerce')

        # Преобразуем в длинный формат
        df_melted = df.melt(
            id_vars=['Country', 'Report Type', 'Category', 'Subcategory', 'Metric'],
            value_vars=year_cols,
            var_name='Year',
            value_name='Value'
        )
        df_melted = df_melted.dropna(subset=['Value'])
        df_melted['Year'] = df_melted['Year'].astype(int)

        # ========================================================
        # АНАЛИЗ ДАТАСЕТА
        # ========================================================
        print("\n" + "=" * 80)
        print(" ДАТАСЕТ УСПЕШНО ЗАГРУЖЕН И ПРОАНАЛИЗИРОВАН ".center(80, "█"))
        print("=" * 80)
        print(f"Размер исходного датасета:     {df.shape[0]:,} строк × {df.shape[1]} столбцов")
        print(f"Размер после melt():           {df_melted.shape[0]:,} строк")
        print(f"Период данных:                 {df_melted['Year'].min()} — {df_melted['Year'].max()} гг.")
        print(f"Количество стран:              {df['Country'].nunique():,} шт.")
        print(f"Пропущенных значений (Value):  {df_melted['Value'].isnull().sum():,} (уже удалены)")
        print("-" * 80)

        # Центральная Азия — отдельно
        ca_present = [c for c in central_asia if c in df['Country'].values]
        print(f"Страны Центральной Азии в датасете ({len(ca_present)} из 5):")
        for c in central_asia:
            if c in df['Country'].values:
                years = sorted(df_melted[df_melted['Country'] == c]['Year'].unique())
                print(f"   ✓ {c:<15} | Годы: {years[0]}–{years[-1]} ({len(years)} лет)")
            else:
                print(f"   ✗ {c:<15} | НЕТ ДАННЫХ")
        print("-" * 80)

        # Report Type
        report_types = df['Report Type'].dropna().unique()
        print(f"Уникальных Report Type: {len(report_types)}")
        print("   Список (топ-10 по частоте):")
        top_reports = df['Report Type'].value_counts().head(10)
        for rt, count in top_reports.items():
            marker = "ВЪЕЗДНОЙ ТУРИЗМ" if 'inbound' in rt.lower() and 'arrival' in rt.lower() else ""
            print(f"   • {rt:<45} → {count:>6} записей  {marker}")

        # какой Report Type используется для въездного туризма
        inbound_candidates = [rt for rt in report_types if 'inbound' in rt.lower() and 'arrival' in rt.lower()]
        print("\n" + "!" * 80)
        print("Основной Report Type для въездного туризма:")
        for rt in inbound_candidates:
            cnt = len(df[df['Report Type'] == rt])
            print(f"   → '{rt}' ← используется в {cnt} записях (ЭТО ТОТ САМЫЙ!)")
        print("!" * 80)

        print(f"\nГотово! Данные загружены и проанализированы.")
        print("Теперь можно запускать: 2 → EDA | 3 → Визуализации | 4 → Дашборд | 5 → Кластеризация")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\nОШИБКА ЗАГРУЗКИ: {e}")
        return False

# ========================================================
# 2. def perform_eda()
# ========================================================
def perform_eda():
    if df is None:
        print("Сначала загрузите данные!")
        return
    print("EDA-АНАЛИЗ")
    print(f"• Стран: {df['Country'].nunique()}")
    print(f"• Годы: {df_melted['Year'].min()}–{df_melted['Year'].max()}")
    missing = df_melted['Value'].isnull().sum()
    total = len(df_melted)
    print(f"• Пропуски: {missing:,} ({missing / total * 100:.1f}%)")

    Q1, Q3 = df_melted['Value'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = ((df_melted['Value'] < Q1 - 1.5 * IQR) | (df_melted['Value'] > Q3 + 1.5 * IQR)).sum()
    print(f"• Выбросы: {outliers:,} ({outliers / len(df_melted) * 100:.1f}%)")


# ========================================================
# 3. def create_visualizations()
# ========================================================
def create_visualizations():
    """3 визуализации по ВСЕМ странам мира"""
    if df_melted is None:
        print("Ошибка: Сначала загрузите данные!")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"\nГенерация ГЛОБАЛЬНЫХ визуализаций → {timestamp}")

    # 1. Heatmap — корреляция по всему миру
    try:
        pivot = df_melted[df_melted['Year'].between(2015, 2020)].pivot_table(
            index='Country', columns='Report Type', values='Value', aggfunc='mean'
        )
        plt.figure(figsize=(12, 9))
        sns.heatmap(pivot.corr(), annot=True, cmap='RdYlBu_r', center=0, fmt='.2f',
                    linewidths=.5, cbar_kws={'shrink': 0.8})
        plt.title('Корреляция типов туристических отчётов (2015–2020) — ВЕСЬ МИР', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/heatmap_global_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Heatmap (все страны) сохранён")
    except Exception as e:
        print(f"Heatmap ошибка: {e}")

    # 2. Boxplot — ТОП-30 стран по среднему количеству туристов
    inbound_all = df_melted[df_melted['Report Type'].str.contains('Inbound', na=False)]
    top30 = inbound_all.groupby('Country')['Value'].mean().nlargest(30).index
    top30_data = inbound_all[inbound_all['Country'].isin(top30)]

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=top30_data, x='Country', y='Value', palette="turbo")
    plt.yscale('log')
    plt.title('ТОП-30 стран по въездному туризму (лог. шкала) — ВЕСЬ МИР', fontsize=16)
    plt.xticks(rotation=60, ha='right')
    plt.ylabel('Количество прибытий (чел.)')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/boxplot_top30_global_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Boxplot ТОП-30 (все страны) сохранён")

    # 3. Line Trend — динамика ТОП-15 стран мира + все 5 стран ЦА
    arrivals = df_melted[
        df_melted['Report Type'].str.contains('Inbound', na=False) &
        df_melted['Category'].str.contains('Arrivals', na=False)
    ].copy()

    yearly = arrivals.groupby(['Country', 'Year'])['Value'].sum().reset_index()
    yearly = yearly[yearly['Value'] > 0]

    # ТОП-15 стран за всё время
    top15_countries = yearly.groupby('Country')['Value'].sum().nlargest(15).index.tolist()
    # все 5 стран ЦА
    ca_must_have = ['KAZAKHSTAN', 'UZBEKISTAN', 'KYRGYZSTAN', 'TAJIKISTAN', 'TURKMENISTAN']
    final_countries = list(set(top15_countries + ca_must_have))

    plot_data = yearly[yearly['Country'].isin(final_countries)]

    fig = px.line(
        plot_data,
        x='Year',
        y='Value',
        color='Country',
        markers=True,
        log_y=True,
        title='Динамика въездного туризма: ТОП-15 стран мира + Центральная Азия (1995–2022)',
        labels={'Value': 'Количество туристов', 'Year': 'Год'},
        height=800,
        color_discrete_sequence=px.colors.qualitative.Vivid + px.colors.qualitative.Safe
    )

    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        legend_title="Страна",
        template="plotly_white",
        hovermode="x unified",
        font=dict(size=13),
        legend=dict(itemsizing="constant")
    )

    # Подсвечиваем страны ЦА жирным
    for country in ca_must_have:
        if country in plot_data['Country'].unique():
            fig.for_each_trace(lambda t: t.update(name=f"<b>{t.name}</b>") if t.name == country else ())

    fig.write_html(f"{RESULTS_DIR}/line_trend_global_top15_plus_ca_{timestamp}.html")
    fig.show()
    print(f"✓ ГЛОБАЛЬНЫЙ ГРАФИК ТРЕНДОВ ГОТОВ → results/line_trend_global_top15_plus_ca_{timestamp}.html")
    print("   • Показаны ТОП-15 стран мира + все 5 стран Центральной Азии")
    print("   • Страны ЦА в легенде выделены жирным")


# ========================================================
# 4. def predict_tourism(country='KAZAKHSTAN')
# ========================================================
def predict_tourism(country='KAZAKHSTAN'):
    global future_predictions
    try:
        # Вариант 1: ищем по Report Type
        data = df_melted[
            (df_melted['Country'] == country) &
            (
                df_melted['Report Type'].str.contains('Inbound', case=False, na=False) |
                df_melted['Category'].str.contains('Arrivals|Visitors', case=False, na=False) |
                df_melted['Metric'].str.contains('arrival|visitor|tourist', case=False, na=False)
            )
        ]

        if data.empty:
            print(f"Нет данных по въездному туризму для {country}")
            return None

        ts = data.groupby('Year')['Value'].sum().reset_index()
        if len(ts) < 3:
            print(f"Слишком мало точек для прогноза: {country}")
            return None

        X = ts[['Year']].values
        y = ts['Value'].values
        model = LinearRegression()
        model.fit(X, y)

        future_years = np.array([[2023], [2024], [2025]])
        future_predictions = model.predict(future_years).round(0).astype(int)

        print(f"ПРОГНОЗ ДЛЯ {country.upper()}:")
        for y, p in zip([2023, 2024, 2025], future_predictions):
            print(f"  {y}: {p:,} туристов")
        return future_predictions

    except Exception as e:
        print(f"Ошибка прогноза: {e}")
        return None


def interactive_dashboard():
    if df_melted is None:
        print("Сначала загрузите данные (пункт 1)!")
        return

    countries = sorted(df_melted['Country'].unique())
    dropdown = widgets.Dropdown(
        options=countries,
        value='KAZAKHSTAN',
        description='Страна:',
        layout=widgets.Layout(width='500px')
    )
    output = widgets.Output()

    def update_plot(_=None):
        with output:
            clear_output(wait=True)
            country = dropdown.value

            data = df_melted[
                (df_melted['Country'] == country) &
                (
                    df_melted['Report Type'].str.contains('Inbound', case=False, na=False) |
                    df_melted['Category'].str.contains('Arrivals|Visitors', case=False, na=False) |
                    df_melted['Metric'].str.contains('arrival|visitor|tourist|non-resident', case=False, na=False)
                )
            ]

            if data.empty:
                print(f"Нет данных по въездному туризму для {country}")
                print("   Попробуйте другую страну (например, FRANCE, SPAIN, CHINA)")
                return

            ts = data.groupby('Year')['Value'].sum().reset_index()

            fig = px.line(
                ts, x='Year', y='Value',
                title=f'Въездной туризм: {country}',
                markers=True,
                height=650,
                labels={'Value': 'Количество туристов', 'Year': 'Год'}
            )
            fig.update_layout(template='plotly_white', hovermode='x unified')

            # Прогноз
            preds = predict_tourism(country)
            if preds is not None:
                fig.add_scatter(
                    x=[2023, 2024, 2025], y=preds,
                    mode='lines+markers',
                    name='Прогноз 2023–2025',
                    line=dict(color='red', width=6, dash='dash'),
                    marker=dict(size=14)
                )
                fig.add_annotation(
                    x=2024, y=preds[1],
                    text="ПРОГНОЗ",
                    showarrow=True,
                    arrowhead=2,
                    bgcolor="red",
                    font=dict(color="white", size=16)
                )

            fig.show()

    dropdown.observe(update_plot, names='value')
    display(widgets.VBox([dropdown, output]))
    update_plot()

# ========================================================
# 5. def run_clustering_and_map()
# ========================================================
def run_clustering_and_map():
    global df_cluster

    if df_melted is None:
        print("Сначала загрузите данные!")
        return

    print("\nВыполняется агрегация и подготовка признаков для кластеризации...")

    # 1. Агрегируем данные по странам
    agg_data = df_melted.groupby('Country').agg(
        Inbound_Total=('Value', lambda x: x[df_melted.loc[x.index, 'Report Type'].str.contains('Inbound', na=False)].sum()),
        Inbound_Mean=('Value', lambda x: x[df_melted.loc[x.index, 'Report Type'].str.contains('Inbound', na=False)].mean()),
        Outbound_Mean=('Value', lambda x: x[df_melted.loc[x.index, 'Report Type'].str.contains('Outbound', na=False)].mean()),
        Expenditure_Mean=('Value', lambda x: x[df_melted.loc[x.index, 'Report Type'].str.contains('Expenditure', na=False)].mean()),
        Employment_Mean=('Value', lambda x: x[df_melted.loc[x.index, 'Metric'].str.contains('employment|jobs', case=False, na=False)].mean())
    ).reset_index()

    # 2. Заполняем пропуски нулями или медианой
    agg_data.fillna(0, inplace=True)

    coords_df = pd.read_csv("country_coords.csv")
    agg_data = agg_data.merge(coords_df, on='Country', how='left')
    agg_data['Lat'] = agg_data['Lat'].fillna(0)
    agg_data['Lon'] = agg_data['Lon'].fillna(0)
    features = ['Inbound_Total', 'Inbound_Mean', 'Outbound_Mean', 'Expenditure_Mean', 'Employment_Mean']
    X_scaled = StandardScaler().fit_transform(agg_data[features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    agg_data['Cluster'] = kmeans.fit_predict(X_scaled)

    df_cluster = agg_data.copy()

    # 5. Сохраняем результат
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    df_cluster.to_csv(f"{RESULTS_DIR}/clusters_{timestamp}.csv", index=False)

    # 6. Создаём карту
    m = folium.Map(location=[43, 65], zoom_start=4, tiles="CartoDB positron")

    marker_cluster = MarkerCluster().add_to(m)

    colors = ['#ff6666', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

    for _, row in df_cluster.iterrows():
        if row['Lat'] == 0 and row['Lon'] == 0:
            continue
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=10 + row['Inbound_Total'] / 1e6,
            popup=folium.Popup(
                f"<b>{row['Country']}</b><br>"
                f"Прибытия: {row['Inbound_Total']:,.0f}<br>"
                f"Кластер: {row['Cluster']}",
                max_width=300
            ),
            color=colors[row['Cluster'] % len(colors)],
            fill=True,
            fillOpacity=0.7
        ).add_to(marker_cluster)

    m.save(f"{RESULTS_DIR}/tourism_map_{timestamp}.html")
    print(f"Кластеризация завершена! Карта: {RESULTS_DIR}/tourism_map_{timestamp}.html")
    print(f"Таблица кластеров: {RESULTS_DIR}/clusters_{timestamp}.csv")
# ========================================================
# 6. def compare_models()
# ========================================================

def compare_models():
    if df_cluster is None:
        print("Сначала выполните пункт 5 (Кластеризация + карта)!")
        return

    print("\nСравнение 4 алгоритмов кластеризации...")

    features = ['Inbound_Total', 'Inbound_Mean', 'Outbound_Mean', 'Expenditure_Mean', 'Employment_Mean']
    X = df_cluster[features].fillna(0)
    X_scaled = StandardScaler().fit_transform(X)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    models = [
        (KMeans(n_clusters=4, random_state=42, n_init=10), "KMeans"),
        (DBSCAN(eps=0.5, min_samples=3), "DBSCAN"),
        (AgglomerativeClustering(n_clusters=4, linkage='ward'), "Agglomerative"),
        (GaussianMixture(n_components=4, random_state=42), "GMM")
    ]

    for model, name in models:
        start = time.time()
        try:
            if name == "DBSCAN":
                labels = model.fit_predict(X_scaled)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_clusters = max(n_clusters, 1)
            else:
                if name == "GMM":
                    labels = model.fit(X_scaled).predict(X_scaled)
                else:
                    labels = model.fit_predict(X_scaled)
                n_clusters = len(set(labels))

            # Метрики только если кластеров больше 1 и нет шума
            if n_clusters > 1 and (name != "DBSCAN" or -1 not in labels or sum(labels == -1) < len(labels) * 0.9):
                sil = round(silhouette_score(X_scaled, labels), 4)
                db = round(davies_bouldin_score(X_scaled, labels), 4)
                ch = round(calinski_harabasz_score(X_scaled, labels), 1)
            else:
                sil, db, ch = "N/A", "N/A", "N/A"
        except Exception as e:
            sil, db, ch, n_clusters = "ERROR", "ERROR", "ERROR", 0

        t = round(time.time() - start, 4)
        results.append({
            "model": name,
            "clusters": n_clusters,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "time_sec": t
        })

    # Сохраняем
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{COMPARISON_DIR}/comparison_{timestamp}.csv", index=False)
    df_res.to_json(f"{COMPARISON_DIR}/comparison_{timestamp}.json", orient="records", indent=2, force_ascii=False)

    print("Сравнение завершено:")
    print(df_res.to_string(index=False))
    print(f"→ Результаты в папке: {COMPARISON_DIR}/")

# ========================================================
# 9. def main_menu()
# ========================================================
def main_menu():
    """Главное меню"""
    print("\n" + "=" * 70)
    print("TOURISM INSIGHTS: CENTRAL ASIA | 16.11.2025 23:53 KZ")
    print("=" * 70)
    while True:
        print("1. Загрузить данные")
        print("2. EDA")
        print("3. Визуализации")
        print("4. Дашборд (все страны + прогноз)")
        print("5. Кластеризация + карта")
        print("6. Сравнение моделей")
        print("7. Выход")
        choice = input("→ ").strip()
        if choice == '1':
            load_data_safely()
        elif choice == '2':
            perform_eda()
        elif choice == '3':
            create_visualizations()
        elif choice == '4':
            interactive_dashboard()
        elif choice == '5':
            run_clustering_and_map()
        elif choice == '6':
            compare_models()
        elif choice == '7':
            break
        else:
            print("Неверный выбор")

# ========================================================
# ЗАПУСК
# ========================================================
if __name__ == "__main__":
    main_menu()