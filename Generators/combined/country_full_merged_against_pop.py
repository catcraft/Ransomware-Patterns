import folium
import requests
import pandas as pd
import json
import numpy as np
from collections import Counter
from datetime import datetime
from ollama import chat
import os

add_markers = False

country_stats = r"\leakmap\Data\geodata\worldpopulation.csv"
# Prefer using this updated file if it already exists
UPDATED_MERGED_PATH = r"\leakmap\Data\combined\merged_updated.csv"
ORIGINAL_MERGED_PATH = r"\leakmap\Final\merged.csv"

# --- Country name to ISO mapping ---
COUNTRY_TO_ISO = {
    'United States': 'USA', 'United Kingdom': 'GBR', 'Germany': 'DEU', 'France': 'FRA', 'Italy': 'ITA', 'Spain': 'ESP',
    'Canada': 'CAN', 'Australia': 'AUS', 'Japan': 'JPN', 'China': 'CHN', 'India': 'IND', 'Brazil': 'BRA', 'Mexico': 'MEX',
    'Netherlands': 'NLD', 'Belgium': 'BEL', 'Switzerland': 'CHE', 'Austria': 'AUT', 'Sweden': 'SWE', 'Norway': 'NOR',
    'Denmark': 'DNK', 'Finland': 'FIN', 'Poland': 'POL', 'Czech Republic': 'CZE', 'Slovakia': 'SVK', 'Hungary': 'HUN',
    'Romania': 'ROU', 'Bulgaria': 'BGR', 'Greece': 'GRC', 'Portugal': 'PRT', 'Ireland': 'IRL', 'Russia': 'RUS', 'Ukraine': 'UKR',
    'Turkey': 'TUR', 'South Africa': 'ZAF', 'Argentina': 'ARG', 'Chile': 'CHL', 'Peru': 'PER', 'Colombia': 'COL',
    'Venezuela': 'VEN', 'Egypt': 'EGY', 'Nigeria': 'NGA', 'Kenya': 'KEN', 'Thailand': 'THA', 'Vietnam': 'VNM', 'Singapore': 'SGP',
    'South Korea': 'KOR', 'New Zealand': 'NZL', 'Saudi Arabia': 'SAU', 'UAE': 'ARE', 'USA': 'USA', 'UK': 'GBR'
}

# Helper: map a country_full to ISO3
def get_iso_for_country_name(country_name: str):
    if not country_name or str(country_name).strip().lower() == 'unknown':
        return None
    return COUNTRY_TO_ISO.get(str(country_name).strip())

# --- Load merged data ---
def load_merged_data(prefer_updated=True):
    if prefer_updated:
        try:
            df = pd.read_csv(UPDATED_MERGED_PATH)
            print(f"Loaded {len(df)} rows from existing merged_updated.csv")
            return df, 'updated'
        except FileNotFoundError:
            print("No existing merged_updated.csv found, will build from merged.csv.")
    try:
        df = pd.read_csv(ORIGINAL_MERGED_PATH)
        print(f"Loaded {len(df)} rows from merged.csv")
        return df, 'original'
    except FileNotFoundError:
        print("Error: merged.csv not found.")
        return pd.DataFrame(), 'none'

# --- Load population data (ISO3 -> population) ---
def load_population_data(path=country_stats):
    try:
        pop_df = pd.read_csv(path)
        # Normalize column names
        pop_df.columns = [c.strip() for c in pop_df.columns]
        # Prefer 2024; else use latest numeric year present
        year_col = '2024' if '2024' in pop_df.columns else None
        if not year_col:
            numeric_years = sorted([int(c) for c in pop_df.columns if c.isdigit()])
            year_col = str(numeric_years[-1]) if numeric_years else None
        if not year_col:
            print("⚠️ Population CSV missing a usable year column.")
            return {}, 'N/A'
        # Keep rows for population indicator if present
        if 'Indicator Name' in pop_df.columns:
            pop_df = pop_df[pop_df['Indicator Name'].astype(str).str.contains('Population', na=False)]
        pop_df = pop_df[['Country Code', year_col]].dropna()
        pop_df[year_col] = pd.to_numeric(pop_df[year_col], errors='coerce')
        pop_df = pop_df.dropna(subset=[year_col])
        # Build ISO3 -> population mapping
        pop_map = dict(zip(pop_df['Country Code'].str.upper(), pop_df[year_col].astype(float)))
        print(f"Loaded population for {len(pop_map)} countries (year {year_col}).")
        return pop_map, year_col
    except Exception as e:
        print(f"⚠️ Error loading population data: {e}")
        return {}, 'N/A'

# --- Ollama helper ---
def query_ollama_for_country(context_text: str):
    prompt = f"""Identify the country (in English) based on this information. Return only the country name or 'unknown'.\n\n{context_text}"""
    try:
        response = chat(messages=[{'role': 'user', 'content': prompt}], model='granite3.1-dense:2b')
        country = response['message']['content'].strip()
        return country if country and country.lower() != 'unknown' else None
    except Exception as e:
        print(f"  ⚠️ Ollama error: {e}")
        return None

# --- Country cleaning ---
def clean_country_name(country):
    if pd.isna(country) or not str(country).strip():
        return None
    mapping = {
        'USA': 'United States', 'US': 'United States', 'UK': 'United Kingdom',
        'Deutschland': 'Germany', 'España': 'Spain', 'Italia': 'Italy', 'Suisse': 'Switzerland', 'Schweiz': 'Switzerland'
    }
    c = str(country).strip()
    return mapping.get(c, c)

# --- Fill missing country info ---
def fill_country_full(df):
    df['country_full'] = None
    for i, row in df.iterrows():
        base = row.get('final_country') or row.get('ollama_country') or None
        country = clean_country_name(base)
        if not country or country.lower() in ['unknown', 'none', '']:
            context = f"Domain: {row.get('domain', '')}\nDescription: {row.get('description', '')}\nCompany: {row.get('company_name', '')}\nLocation: {row.get('location', '')}"
            print(f"→ Inferring country for {row.get('domain', 'unknown domain')} ...")
            country = query_ollama_for_country(context)
        if not country:
            tld = str(row.get('tld', '')).lower()
            fallback = COUNTRY_TO_ISO.get(tld.upper(), None)
            if fallback:
                country = fallback
        if not country:
            country = 'Unknown'
        df.at[i, 'country_full'] = clean_country_name(country)
    print("✓ country_full column filled")
    return df

# --- Count leaks by country ---
def count_leaks_by_country(df):
    counts = Counter()
    for _, row in df.iterrows():
        c = row.get('country_full')
        if c and c != 'Unknown':
            counts[c] += 1
    return counts

# --- GeoJSON Loader ---
def download_world_geojson():
    try:
        with open(r'\leakmap\Data\geodata\geojson.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ GeoJSON load failed: {e}")
        return {'type': 'FeatureCollection', 'features': []}

# --- Map (per capita) ---
def create_country_choropleth_map(country_counts, population_by_iso, pop_year, output_file='leak_map_countries.html'):
    geo = download_world_geojson()
    rows = []
    # Build rows with per-capita metric
    for country_name, cnt in country_counts.items():
        iso = get_iso_for_country_name(country_name) or country_name.upper()[:3]
        pop = population_by_iso.get(iso)
        per_million = (cnt / pop * 1_000_000.0) if pop and pop > 0 else None
        rows.append({
            'country': country_name,
            'iso': iso,
            'count': cnt,
            'population': pop,
            'per_million': per_million
        })
    df = pd.DataFrame(rows)
    # Determine max rate for severity scaling
    valid_rates = df['per_million'].dropna()
    max_rate = float(valid_rates.max()) if not valid_rates.empty else 1.0
    df['severity'] = df['per_million'].apply(lambda v: get_severity_level(v if pd.notna(v) else 0, max_rate))
    # Base map
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='OpenStreetMap')
    # Choropleth by leaks per million
    folium.Choropleth(
        geo_data=geo,
        name='Leaks per million (Population-adjusted)',
        data=df.dropna(subset=['per_million']),
        columns=['iso', 'per_million'],
        key_on='feature.id',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f'Leaks per million residents ({pop_year})',
        nan_fill_color='lightgray',
        nan_fill_opacity=0.3
    ).add_to(m)
    # Pins
    for _, row in df.iterrows():
        coords = get_country_centroid(geo, row['iso'])
        if not coords:
            continue
        color = get_severity_color(row['severity'])
        rate = row['per_million']
        pop_txt = f"{int(row['population']):,}" if pd.notna(row['population']) else "n/a"
        rate_txt = f"{rate:,.2f}" if pd.notna(rate) else "n/a"
        radius = 7 if not pd.notna(rate) else min(5 + (rate / max_rate) * 20, 25)
        if add_markers:
            folium.CircleMarker(
                location=coords,
                radius=radius,
                popup=folium.Popup(
                    f"""
                    <div style="font-family: Arial; width: 220px;">
                        <h4 style="margin: 0; color: #333;">{row['country']}</h4>
                        <hr style="margin: 5px 0;">
                        <p style="margin: 5px 0;"><b>Leaked Domains:</b> {row['count']}</p>
                        <p style="margin: 5px 0;"><b>Population ({pop_year}):</b> {pop_txt}</p>
                        <p style="margin: 5px 0;"><b>Leaks per million:</b> {rate_txt}</p>
                        <p style="margin: 5px 0;"><b>Severity:</b> {row['severity']}</p>
                        <p style="margin: 5px 0;"><b>ISO Code:</b> {row['iso']}</p>
                    </div>
                    """,
                    max_width=260
                ),
                color='black',
                fillColor=color,
                fillOpacity=0.7,
                weight=2,
                tooltip=f"{row['country']}: {rate_txt} per million"
            ).add_to(m)
    # Legend (relative to max rate)
    legend_html = create_custom_legend(max_rate=max_rate,
                                       total_leaks=int(df['count'].sum()),
                                       countries_count=len(country_counts),
                                       metric_label=f"Leaks per million ({pop_year})")
    m.get_root().html.add_child(folium.Element(legend_html))
    # Title
    title_html = f'''
    <h2 align="center" style="font-size:24px; margin-top:10px; color:#333;">
        <b>Global Data Leaks per Million Residents</b>
    </h2>
    <p align="center" style="font-size:14px; color:#666; margin-top:5px;">
        Countries colored by leaks per million residents ({pop_year})
    </p>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    # Layers
    folium.LayerControl().add_to(m)
    # Save
    if output_file:
        if add_markers:
            output_file = output_file.replace('.html', '_with_markers.html')
    m.save(output_file)
    print(f"✓ Choropleth map saved as {output_file}")
    # Stats
    print_map_statistics(country_counts, len(df), per_million_series=df.set_index('country')['per_million'])

def get_country_centroid(geojson, iso_code):
    """Get approximate centroid of a country feature for marker placement"""
    try:
        for feature in geojson.get('features', []):
            if feature.get('id') == iso_code:
                geometry = feature.get('geometry', {})
                if geometry.get('type') == 'Polygon':
                    coords = geometry['coordinates'][0]
                    if coords:
                        lats = [coord[1] for coord in coords]
                        lngs = [coord[0] for coord in coords]
                        return [sum(lats)/len(lats), sum(lngs)/len(lngs)]
                elif geometry.get('type') == 'MultiPolygon':
                    coords = geometry['coordinates'][0][0]
                    if coords:
                        lats = [coord[1] for coord in coords]
                        lngs = [coord[0] for coord in coords]
                        return [sum(lats)/len(lats), sum(lngs)/len(lngs)]
                break
    except Exception:
        pass
    return None

def get_severity_level(value, max_value):
    """Determine severity level based on relative value vs. max"""
    if not value or max_value <= 0:
        return 'No Data'
    ratio = value / max_value
    if ratio <= 0.10:
        return 'Low'
    elif ratio <= 0.30:
        return 'Medium'
    elif ratio <= 0.60:
        return 'High'
    else:
        return 'Critical'

def get_severity_color(severity):
    """Get color for severity level"""
    colors = {
        'Critical': '#800026',
        'High': '#BD0026',
        'Medium': '#E31A1C',
        'Low': '#FC4E2A',
        'No Data': 'lightgray'
    }
    return colors.get(severity, 'blue')

def create_custom_legend(max_rate, total_leaks, countries_count, metric_label):
    """Create a toggleable custom HTML legend for per-capita metric"""
    return f'''
    <div id="legendContainer" style="position: fixed; 
                top: 10px; right: 10px; width: 240px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px; border-radius: 5px; 
                box-shadow: 0 0 15px rgba(0,0,0,0.2);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0; color: #333;">{metric_label}</h4>
            <button onclick="toggleLegend()" style="border:none; background:none; font-size:16px; cursor:pointer;">✖</button>
        </div>
        <div id="legendContent" style="margin-top:10px;">
            <div style="margin: 5px 0;"><i style="background: #800026; width: 12px; height: 12px; display: inline-block; margin-right: 8px;"></i>Critical (≥60% of max)</div>
            <div style="margin: 5px 0;"><i style="background: #BD0026; width: 12px; height: 12px; display: inline-block; margin-right: 8px;"></i>High (30–60%)</div>
            <div style="margin: 5px 0;"><i style="background: #E31A1C; width: 12px; height: 12px; display: inline-block; margin-right: 8px;"></i>Medium (10–30%)</div>
            <div style="margin: 5px 0;"><i style="background: #FC4E2A; width: 12px; height: 12px; display: inline-block; margin-right: 8px;"></i>Low (≤10%)</div>
            <div style="margin: 5px 0;"><i style="background: lightgray; width: 12px; height: 12px; display: inline-block; margin-right: 8px;"></i>No Data</div>
            <hr style="margin: 8px 0;">
            <small><b>Total leaks:</b> {total_leaks:,}<br><b>Countries:</b> {countries_count}<br><b>Max rate:</b> {max_rate:,.2f} per M</small>
        </div>
    </div>
    <button id="showLegendButton" onclick="toggleLegend()" 
            style="display:none; position: fixed; top: 10px; right: 10px; 
                   background-color: white; border: 2px solid grey; border-radius: 5px;
                   padding: 5px 10px; cursor: pointer; z-index:9999;">
        Show Legend
    </button>
    <script>
    function toggleLegend() {{
        var legend = document.getElementById('legendContainer');
        var showButton = document.getElementById('showLegendButton');
        if (legend.style.display === 'none') {{
            legend.style.display = 'block';
            showButton.style.display = 'none';
        }} else {{
            legend.style.display = 'none';
            showButton.style.display = 'block';
        }}
    }}
    </script>
    '''

def print_map_statistics(country_counts, mapped_countries, per_million_series=None):
    """Print statistics about the map"""
    total_leaks = sum(country_counts.values())
    total_countries = len(country_counts)
    print(f"\n=== CHOROPLETH MAP STATISTICS ===")
    print(f"Total leaked domains: {total_leaks}")
    print(f"Countries with leaks: {total_countries}")
    print(f"Countries successfully mapped: {mapped_countries}")
    print(f"Average leaks per country: {total_leaks / total_countries:.1f}")
    print(f"\n=== TOP 10 MOST AFFECTED COUNTRIES (BY COUNT) ===")
    for i, (country, count) in enumerate(country_counts.most_common(10), 1):
        percentage = (count / total_leaks) * 100 if total_leaks else 0
        print(f"{i:2d}. {country:<25} {count:3d} leaks ({percentage:4.1f}%)")
    if per_million_series is not None:
        pm = per_million_series.dropna().sort_values(ascending=False).head(10)
        print(f"\n=== TOP 10 BY LEAKS PER MILLION ===")
        for i, (country, rate) in enumerate(pm.items(), 1):
            print(f"{i:2d}. {country:<25} {rate:10.2f} per M")

# --- Main ---
def main():
    df, src = load_merged_data(prefer_updated=True)
    if df.empty:
        return

    # If using pre-generated file but it's missing country_full, rebuild it
    needs_inference = (src != 'updated') or ('country_full' not in df.columns or df['country_full'].isna().all())

    if needs_inference:
        df = fill_country_full(df)
        os.makedirs(os.path.dirname(UPDATED_MERGED_PATH), exist_ok=True)
        df.to_csv(UPDATED_MERGED_PATH, index=False)
        print(f"✓ Saved updated CSV as {UPDATED_MERGED_PATH}")
    else:
        print("✓ Using pre-generated merged_updated.csv; skipping country inference.")

    country_counts = count_leaks_by_country(df)
    print(f"Found {len(country_counts)} countries with leaks.")

    # Load population and render per-capita map
    population_by_iso, pop_year = load_population_data(country_stats)
    create_country_choropleth_map(country_counts, population_by_iso, pop_year)

if __name__ == '__main__':
    main()
