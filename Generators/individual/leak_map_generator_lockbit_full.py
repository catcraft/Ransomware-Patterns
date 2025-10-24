import folium
import requests
import pandas as pd
from collections import Counter
import json
import numpy as np

# Country name to ISO code mapping for common countries in leak data
COUNTRY_TO_ISO = {
    'United States': 'USA',
    'United Kingdom': 'GBR', 
    'Germany': 'DEU',
    'France': 'FRA',
    'Italy': 'ITA',
    'Spain': 'ESP',
    'Canada': 'CAN',
    'Australia': 'AUS',
    'Japan': 'JPN',
    'China': 'CHN',
    'India': 'IND',
    'Brazil': 'BRA',
    'Mexico': 'MEX',
    'Netherlands': 'NLD',
    'Belgium': 'BEL',
    'Switzerland': 'CHE',
    'Austria': 'AUT',
    'Sweden': 'SWE',
    'Norway': 'NOR',
    'Denmark': 'DNK',
    'Finland': 'FIN',
    'Poland': 'POL',
    'Czech Republic': 'CZE',
    'Slovakia': 'SVK',
    'Hungary': 'HUN',
    'Romania': 'ROU',
    'Bulgaria': 'BGR',
    'Greece': 'GRC',
    'Portugal': 'PRT',
    'Ireland': 'IRL',
    'Iceland': 'ISL',
    'Luxembourg': 'LUX',
    'Slovenia': 'SVN',
    'Croatia': 'HRV',
    'Estonia': 'EST',
    'Latvia': 'LVA',
    'Lithuania': 'LTU',
    'Russia': 'RUS',
    'Ukraine': 'UKR',
    'Turkey': 'TUR',
    'Israel': 'ISR',
    'Saudi Arabia': 'SAU',
    'United Arab Emirates': 'ARE',
    'South Africa': 'ZAF',
    'Egypt': 'EGY',
    'Nigeria': 'NGA',
    'Kenya': 'KEN',
    'Morocco': 'MAR',
    'Algeria': 'DZA',
    'Tunisia': 'TUN',
    'Libya': 'LBY',
    'Ethiopia': 'ETH',
    'Ghana': 'GHA',
    'South Korea': 'KOR',
    'North Korea': 'PRK',
    'Thailand': 'THA',
    'Vietnam': 'VNM',
    'Malaysia': 'MYS',
    'Singapore': 'SGP',
    'Indonesia': 'IDN',
    'Philippines': 'PHL',
    'New Zealand': 'NZL',
    'Argentina': 'ARG',
    'Chile': 'CHL',
    'Peru': 'PER',
    'Colombia': 'COL',
    'Venezuela': 'VEN',
    'Ecuador': 'ECU',
    'Bolivia': 'BOL',
    'Uruguay': 'URY',
    'Paraguay': 'PRY',
    'Guyana': 'GUY',
    'Suriname': 'SUR',
    'Afghanistan': 'AFG',
    'Pakistan': 'PAK',
    'Bangladesh': 'BGD',
    'Sri Lanka': 'LKA',
    'Nepal': 'NPL',
    'Bhutan': 'BTN',
    'Myanmar': 'MMR',
    'Cambodia': 'KHM',
    'Laos': 'LAO',
    'Mongolia': 'MNG',
    'Kazakhstan': 'KAZ',
    'Uzbekistan': 'UZB',
    'Turkmenistan': 'TKM',
    'Kyrgyzstan': 'KGZ',
    'Tajikistan': 'TJK',
    'Iran': 'IRN',
    'Iraq': 'IRQ',
    'Syria': 'SYR',
    'Lebanon': 'LBN',
    'Jordan': 'JOR',
    'Kuwait': 'KWT',
    'Qatar': 'QAT',
    'Bahrain': 'BHR',
    'Oman': 'OMN',
    'Yemen': 'YEM',
    'Georgia': 'GEO',
    'Armenia': 'ARM',
    'Azerbaijan': 'AZE',
    'Belarus': 'BLR',
    'Moldova': 'MDA',
    'Serbia': 'SRB',
    'Montenegro': 'MNE',
    'Bosnia and Herzegovina': 'BIH',
    'North Macedonia': 'MKD',
    'Albania': 'ALB',
    'Kosovo': 'XKX',
    'Malta': 'MLT',
    'Cyprus': 'CYP',
    'Seychelles': 'SYC',
    'Mauritius': 'MUS',
    'Madagascar': 'MDG',
    'Botswana': 'BWA',
    'Namibia': 'NAM',
    'Zambia': 'ZMB',
    'Zimbabwe': 'ZWE',
    'Mozambique': 'MOZ',
    'Tanzania': 'TZA',
    'Uganda': 'UGA',
    'Rwanda': 'RWA',
    'Burundi': 'BDI',
    'Malawi': 'MWI',
    'USA': 'USA',
    'UK': 'GBR',
    'Trinidad': 'TTO',
}

def download_world_geojson():
    with open('geojson.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def create_fallback_geojson():
    """Create a minimal GeoJSON for major countries if download fails"""
    # This is a very basic fallback - in production, you'd want to include a local GeoJSON file
    return {
        "type": "FeatureCollection",
        "features": []
    }

def load_leak_data(csv_file='leak_results.csv'):
    """Load leak data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        return df
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Please run the main leak analyzer first.")
        return pd.DataFrame()

def count_leaks_by_country(df):
    """Count leaks by country from the dataframe"""
    if df.empty:
        return Counter()
    
    country_counts = Counter()
    for _, row in df.iterrows():
        country = row['final_country']
        # Clean up country names and handle special cases
        country = clean_country_name(country)
        country_counts[country] += 1
    
    return country_counts

def clean_country_name(country):
    """Clean and standardize country names"""
    if pd.isna(country):
        return 'Unknown'
    
    country = str(country).strip()
    
    # Handle special cases
    if country.startswith('Unknown ('):
        return 'Unknown'
    
    # Map common variations to standard names
    mapping = {
        'United States of America': 'United States',
        'USA': 'United States',
        'US': 'United States',
        'UK': 'United Kingdom',
        'Great Britain': 'United Kingdom',
        'Deutschland': 'Germany',
        'Italia': 'Italy',
        'España': 'Spain',
        'République française': 'France',
        'Nederland': 'Netherlands',
        'Schweiz': 'Switzerland',
        'Österreich': 'Austria',
        'Sverige': 'Sweden',
        'Norge': 'Norway',
        'Danmark': 'Denmark',
        'Suomi': 'Finland',
        'Polska': 'Poland',
        'Česká republika': 'Czech Republic',
        'Slovensko': 'Slovakia',
        'Magyarország': 'Hungary',
        'România': 'Romania',
        'България': 'Bulgaria',
        'Ελλάδα': 'Greece',
        'Россия': 'Russia',
        'Україна': 'Ukraine',
        'Türkiye': 'Turkey',
    }
    
    return mapping.get(country, country)

def create_country_choropleth_map(country_counts, output_file='leak_map_countries.html'):
    """Create a choropleth map that colors entire countries based on leak counts"""
    
    print("Downloading world country boundaries...")
    world_geojson = download_world_geojson()
    
    if not world_geojson.get('features'):
        print("No GeoJSON data available. Creating a basic map with markers...")
        return create_fallback_marker_map(country_counts, output_file)
    
    # Prepare data for choropleth
    country_data = []
    max_count = max(country_counts.values()) if country_counts else 1
    
    for country, count in country_counts.items():
        iso_code = COUNTRY_TO_ISO.get(country, country.upper()[:3])
        country_data.append({
            'country': country,
            'iso_code': iso_code,
            'leak_count': count,
            'severity': get_severity_level(count, max_count)
        })
    
    # Create dataframe for choropleth
    df_map = pd.DataFrame(country_data)
    
    # Normalize leak counts logarithmically
    df_map['log_leak_count'] = df_map['leak_count'].apply(lambda x: np.log10(x + 1))

    # Create base map
    m = folium.Map(
        location=[20, 0], 
        zoom_start=2,
        tiles='OpenStreetMap'
    )
    

    folium.Choropleth(
        geo_data=world_geojson,
        name='Data Leaks by Country',
        data=df_map,
        columns=['iso_code', 'log_leak_count'],  # <-- use log scale
        key_on='feature.id',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Number of Data Leaks (log scale)',
        nan_fill_color='lightgray',
        nan_fill_opacity=0.3
    ).add_to(m)
    
    # Add tooltips with detailed information
    for _, row in df_map.iterrows():
        # Find matching feature in GeoJSON for tooltip placement
        for feature in world_geojson['features']:
            if feature.get('id') == row['iso_code']:
                # Get country centroid for tooltip placement (simplified)
                coords = get_country_centroid(feature)
                if coords:
                    folium.CircleMarker(
                        location=coords,
                        radius=3,
                        popup=folium.Popup(
                            f"""
                            <div style="font-family: Arial; width: 200px;">
                                <h4 style="margin: 0; color: #333;">{row['country']}</h4>
                                <hr style="margin: 5px 0;">
                                <p style="margin: 5px 0;"><b>Leaked Domains:</b> {row['leak_count']}</p>
                                <p style="margin: 5px 0;"><b>Severity:</b> {row['severity']}</p>
                                <p style="margin: 5px 0;"><b>ISO Code:</b> {row['iso_code']}</p>
                            </div>
                            """,
                            max_width=220
                        ),
                        color='red',
                        fillColor='red',
                        fillOpacity=0.8,
                        weight=1,
                        tooltip=f"{row['country']}: {row['leak_count']} leaks"
                    ).add_to(m)
                break
    
    # Add custom legend
    legend_html = create_custom_legend(country_counts, max_count)
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = '''
    <h2 align="center" style="font-size:24px; margin-top:10px; color:#333;">
        <b>Global Data Leaks by Country (Lockbit 2.0) (RU)</b>
    </h2>
    <p align="center" style="font-size:14px; color:#666; margin-top:5px;">
        Countries colored by number of leaked domains
    </p>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    m.save(output_file)
    print(f"Choropleth map saved as {output_file}")
    
    # Print statistics
    print_map_statistics(country_counts, len(df_map))
    
    return m

def get_country_centroid(feature):
    """Get approximate centroid of a country feature for marker placement"""
    try:
        # This is a simplified centroid calculation
        # In production, you'd use proper geometric calculations
        geometry = feature.get('geometry', {})
        if geometry.get('type') == 'Polygon':
            coords = geometry['coordinates'][0]
            if coords:
                lats = [coord[1] for coord in coords]
                lngs = [coord[0] for coord in coords]
                return [sum(lats)/len(lats), sum(lngs)/len(lngs)]
        elif geometry.get('type') == 'MultiPolygon':
            # Use first polygon for simplicity
            coords = geometry['coordinates'][0][0]
            if coords:
                lats = [coord[1] for coord in coords]
                lngs = [coord[0] for coord in coords]
                return [sum(lats)/len(lats), sum(lngs)/len(lngs)]
    except Exception:
        pass
    return None

def get_severity_level(count, max_count):
    """Determine severity level based on count"""
    if count == 0:
        return 'No Data'
    elif count <= max_count * 0.1:
        return 'Low'
    elif count <= max_count * 0.3:
        return 'Medium'
    elif count <= max_count * 0.6:
        return 'High'
    else:
        return 'Critical'

def create_custom_legend(country_counts, max_count):
    """Create a toggleable custom HTML legend"""
    total_leaks = sum(country_counts.values())
    
    return f'''
    <div id="legendContainer" style="position: fixed; 
                top: 10px; right: 10px; width: 220px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px; border-radius: 5px; 
                box-shadow: 0 0 15px rgba(0,0,0,0.2);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h4 style="margin: 0; color: #333;">Data Leak Intensity</h4>
            <button onclick="toggleLegend()" style="border:none; background:none; font-size:16px; cursor:pointer;">✖</button>
        </div>
        <div id="legendContent" style="margin-top:10px;">
            <div style="margin: 5px 0;"><i style="background: #800026; width: 12px; height: 12px; display: inline-block; margin-right: 8px;"></i>Critical (60%+)</div>
            <div style="margin: 5px 0;"><i style="background: #BD0026; width: 12px; height: 12px; display: inline-block; margin-right: 8px;"></i>High (30–60%)</div>
            <div style="margin: 5px 0;"><i style="background: #E31A1C; width: 12px; height: 12px; display: inline-block; margin-right: 8px;"></i>Medium (10–30%)</div>
            <div style="margin: 5px 0;"><i style="background: #FC4E2A; width: 12px; height: 12px; display: inline-block; margin-right: 8px;"></i>Low (0–10%)</div>
            <div style="margin: 5px 0;"><i style="background: lightgray; width: 12px; height: 12px; display: inline-block; margin-right: 8px;"></i>No Data</div>
            <hr style="margin: 8px 0;">
            <small><b>Total:</b> {total_leaks} leaks<br><b>Countries:</b> {len(country_counts)}</small>
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


def create_fallback_marker_map(country_counts, output_file):
    """Create a fallback map with markers if GeoJSON fails"""
    from Utils.country_mapping import COUNTRY_COORDINATES
    
    m = folium.Map(location=[20, 0], zoom_start=2)
    max_count = max(country_counts.values()) if country_counts else 1
    
    for country, count in country_counts.items():
        if country in COUNTRY_COORDINATES:
            coords = COUNTRY_COORDINATES[country]
            severity = get_severity_level(count, max_count)
            
            color = {
                'Critical': 'darkred',
                'High': 'red', 
                'Medium': 'orange',
                'Low': 'green',
                'No Data': 'gray'
            }.get(severity, 'blue')
            
            folium.CircleMarker(
                location=[coords['lat'], coords['lng']],
                radius=min(5 + (count / max_count) * 20, 25),
                popup=f"<b>{country}</b><br>Leaks: {count}<br>Severity: {severity}",
                color='black',
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
    
    m.save(output_file)
    print(f"Fallback marker map saved as {output_file}")
    return m

def print_map_statistics(country_counts, mapped_countries):
    """Print statistics about the map"""
    total_leaks = sum(country_counts.values())
    total_countries = len(country_counts)
    
    print(f"\n=== CHOROPLETH MAP STATISTICS ===")
    print(f"Total leaked domains: {total_leaks}")
    print(f"Countries with leaks: {total_countries}")
    print(f"Countries successfully mapped: {mapped_countries}")
    print(f"Average leaks per country: {total_leaks / total_countries:.1f}")
    
    print(f"\n=== TOP 10 MOST AFFECTED COUNTRIES ===")
    for i, (country, count) in enumerate(country_counts.most_common(10), 1):
        percentage = (count / total_leaks) * 100
        print(f"{i:2d}. {country:<25} {count:3d} leaks ({percentage:4.1f}%)")

def main():
    """Main function to create the choropleth map"""
    print("Creating Data Leak Choropleth Map...")
    
    # Load leak data
    print("Loading leak data...")
    df = load_leak_data()
    
    if df.empty:
        print("No data found. Please run the main leak analyzer first.")
        return
    
    # Count leaks by country
    country_counts = count_leaks_by_country(df)
    
    if not country_counts:
        print("No country data found.")
        return
    
    # Create choropleth map
    create_country_choropleth_map(country_counts)
    
    print("\nDone! Open 'leak_map_countries.html' in your browser to view the choropleth map.")

if __name__ == "__main__":
    main()
