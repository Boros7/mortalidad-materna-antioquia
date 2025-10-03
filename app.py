# app.py - Dash app final y robusto para Mortalidad Materna (Antioquia)
# Requisitos: dash, pandas, numpy, plotly
# Archivos necesarios en la misma carpeta:
# - Mortalidad_de_maternas_en_el_departamento_de_Antioquia_desde_2005_20250915.csv
# - geojson_munis_simplified.geojson

import os, json, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_MORTALIDAD = os.path.join(BASE_DIR, "Mortalidad_de_maternas_en_el_departamento_de_Antioquia_desde_2005_20250915.csv")
GEOJSON_PATH = os.path.join(BASE_DIR, "geojson_munis_simplified.geojson")

# -------------------------
# CARGA Y PREPARACIÓN
# -------------------------
print("📥 Cargando datos...")
df = pd.read_csv(CSV_MORTALIDAD, encoding="utf-8")

df["NumeroCasos"] = pd.to_numeric(df.get("NumeroCasos", 0), errors="coerce").fillna(0).astype(int)
df["NumeroPoblacionObjetivo"] = pd.to_numeric(df.get("NumeroPoblacionObjetivo", 0), errors="coerce").fillna(0).astype(int)
df["CodigoMunicipio"] = df["CodigoMunicipio"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(5)
df["NombreRegion"] = df["NombreRegion"].fillna("Sin región")

# crear tasa
df["tasa_100k"] = np.nan
mask = (df["NumeroPoblacionObjetivo"] > 0)
df.loc[mask, "tasa_100k"] = (df.loc[mask, "NumeroCasos"] / df.loc[mask, "NumeroPoblacionObjetivo"]) * 100000

print(f" - Registros CSV: {len(df)}")
years = sorted(pd.unique(df["Año"]))

# Agregados históricos por municipio (para el mapa promedio)
muni_agg = (
    df.groupby("CodigoMunicipio", as_index=False)
      .agg(NumeroCasos=("NumeroCasos","sum"),
           NumeroPoblacionObjetivo=("NumeroPoblacionObjetivo","sum"),
           NombreMunicipio=("NombreMunicipio","first"),
           NombreRegion=("NombreRegion","first"))
)
muni_agg["tasa_100k"] = (muni_agg["NumeroCasos"] / muni_agg["NumeroPoblacionObjetivo"].replace({0: np.nan})) * 100000
muni_agg["CodigoMunicipio"] = muni_agg["CodigoMunicipio"].astype(str).str.zfill(5)

vals = muni_agg["tasa_100k"].replace([np.inf, -np.inf], np.nan).dropna()
global_vmax = float(vals.quantile(0.95)) if len(vals)>0 else None

# cargar geojson
geojson_data = {}
available_ids = set()
if os.path.exists(GEOJSON_PATH):
    with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
    feats = geojson_data.get("features", [])
    for feat in feats:
        props = feat.get("properties", {}) or {}
        if not feat.get("id"):
            code = props.get("CodigoMunicipio") or props.get("codigo")
            if code:
                feat["id"] = str(code).zfill(5)
    available_ids = {str(feat.get("id")).zfill(5) for feat in feats if feat.get("id")}
    print(" - GeoJSON cargado con", len(available_ids), "ids")
else:
    print("⚠️ No encontré geojson en:", GEOJSON_PATH)

# -------------------------
# DASH APP
# -------------------------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("📊 Mortalidad Materna — Antioquia (2005–2024)"),
    html.Div([
        dcc.Dropdown(id="year-dropdown",
                     options=[{"label": str(int(y)), "value": int(y)} for y in years],
                     value=int(max(years)), clearable=False,
                     style={"width":"200px"}),
        dcc.Dropdown(id="region-dropdown",
                     options=[{"label":"Todas","value":"all"}] + [{"label":r,"value":r} for r in sorted(df["NombreRegion"].unique())],
                     value="all", clearable=False,
                     style={"width":"300px","marginLeft":"12px"})
    ], style={"display":"flex","gap":"12px","marginBottom":"12px"}),

    dcc.Graph(id="map-graph", style={"height":"650px"}),

    html.Div(style={"display":"flex","gap":"16px"}, children=[
        dcc.Graph(id="time-series-graph", style={"flex":"1"}),
        dcc.Graph(id="distribution-graph", style={"flex":"1"})
    ]),

    html.Div(style={"display":"flex","gap":"16px","marginTop":"12px"}, children=[
        dcc.Graph(id="region-boxplot", style={"flex":"1"}),
        dcc.Graph(id="scatter-plot", style={"flex":"1"})
    ]),

    html.Div(id="stats-summary", style={"marginTop":"12px","padding":"10px","backgroundColor":"#f7fbff"})
], style={"padding":"10px"})

# -------------------------
# CALLBACK
# -------------------------
@app.callback(
    [Output('map-graph','figure'),
     Output('time-series-graph','figure'),
     Output('distribution-graph','figure'),
     Output('region-boxplot','figure'),
     Output('scatter-plot','figure'),
     Output('stats-summary','children')],
    [Input('year-dropdown','value'),
     Input('region-dropdown','value')]
)
def update_dashboard(selected_year, selected_region):
    try:
        # filtros
        filtered_df = df if selected_region == "all" else df[df["NombreRegion"] == selected_region]
        year_df = filtered_df[filtered_df["Año"] == int(selected_year)]

        # mapa (igual que antes)
        map_data = muni_agg.copy()
        if available_ids:
            map_data = map_data[map_data["CodigoMunicipio"].isin(available_ids)].copy()
        z_vals = map_data["tasa_100k"].fillna(0).tolist()
        locations = map_data["CodigoMunicipio"].astype(str).tolist()
        zmax = float(global_vmax) if (global_vmax and global_vmax>0) else None

        chor = go.Choroplethmapbox(
            geojson=geojson_data, locations=locations, z=z_vals,
            featureidkey="id", colorscale="Viridis", zmin=0, zmax=zmax,
            marker_line_width=0.4, marker_line_color="black",
            colorbar=dict(title="Tasa x100k"),
            hovertemplate="<b>%{customdata[0]}</b><br>Código: %{location}<br>Tasa: %{customdata[1]:.2f}<extra></extra>",
            customdata=np.stack([map_data["NombreMunicipio"].astype(str), map_data["tasa_100k"].fillna(0)], axis=-1)
        )
        map_fig = go.Figure(data=[chor], layout=go.Layout(
            mapbox=dict(style="open-street-map", center={"lat":6.5,"lon":-75.5}, zoom=6.6),
            margin={"r":0,"t":0,"l":0,"b":0}
        ))

        # -------------------------
        # GRÁFICAS
        # -------------------------

        # Serie temporal (línea y barras en doble eje)
        temporal_data = filtered_df.groupby("Año", as_index=False).agg(
            NumeroCasos=("NumeroCasos","sum"),
            NumeroPoblacionObjetivo=("NumeroPoblacionObjetivo","sum")
        )
        temporal_data["tasa_100k"] = (temporal_data["NumeroCasos"] / temporal_data["NumeroPoblacionObjetivo"].replace({0:np.nan})) * 100000

        time_fig = go.Figure()
        time_fig.add_trace(go.Scatter(
            x=temporal_data["Año"], y=temporal_data["tasa_100k"],
            mode="lines+markers", name="Tasa x100k",
            line=dict(color="royalblue", width=2)
        ))
        time_fig.add_trace(go.Bar(
            x=temporal_data["Año"], y=temporal_data["NumeroCasos"],
            name="Casos", marker=dict(color="indianred"), opacity=0.6, yaxis="y2"
        ))
        time_fig.update_layout(
            title="Evolución anual de tasa y casos",
            xaxis=dict(title="Año"),
            yaxis=dict(title="Tasa por 100k"),
            yaxis2=dict(title="Casos", overlaying="y", side="right"),
            legend=dict(orientation="h", y=-0.2),
            template="plotly_white"
        )

        # Histograma año seleccionado
        dist_fig = px.histogram(year_df, x="tasa_100k", nbins=20, title="Distribución tasas (año seleccionado)")
        dist_fig.update_layout(xaxis_title="Tasa por 100k", yaxis_title="N° municipios", template="plotly_white")

        # Boxplot por región
        box_fig = px.box(filtered_df, x="NombreRegion", y="tasa_100k", points="all",
                         title="Distribución por región")
        box_fig.update_layout(yaxis_title="Tasa por 100k", xaxis_title="Región", template="plotly_white")

        # Scatter Población vs Tasa (año seleccionado)
        scatter_fig = px.scatter(year_df, x="NumeroPoblacionObjetivo", y="tasa_100k",
                                 hover_name="NombreMunicipio", size="NumeroCasos",
                                 title="Población vs Tasa (año seleccionado)",
                                 log_x=True)
        scatter_fig.update_layout(xaxis_title="Población objetivo (log)", yaxis_title="Tasa por 100k", template="plotly_white")

        # -------------------------
        # STATS
        # -------------------------
        total_casos = int(filtered_df["NumeroCasos"].sum())
        total_pob = int(filtered_df["NumeroPoblacionObjetivo"].sum())
        tasa_prom = (total_casos/total_pob)*100000 if total_pob>0 else np.nan
        municipios_afectados = int((filtered_df["NumeroCasos"]>0).sum())
        total_mun = int(filtered_df["CodigoMunicipio"].nunique())

        stats_html = html.Div([
            html.P(f"Total casos (filtro): {total_casos}"),
            html.P(f"Tasa promedio (filtro): {tasa_prom:.2f} x100k" if np.isfinite(tasa_prom) else "Tasa promedio: N/A"),
            html.P(f"Municipios afectados: {municipios_afectados}/{total_mun}"),
            html.P(f"Año: {selected_year} | Región: {'Todas' if selected_region=='all' else selected_region}")
        ])

        return map_fig, time_fig, dist_fig, box_fig, scatter_fig, stats_html

    except Exception as e:
        print("[ERROR]", e)
        empty = go.Figure(); empty.update_layout(title="Error interno: "+str(e))
        return empty, empty, empty, empty, empty, html.Div([html.P("Error interno: "+str(e))])

if __name__ == "__main__":
    print("Servidor local: http://127.0.0.1:8050")
    app.run(debug=True, host="0.0.0.0", port=8050)
