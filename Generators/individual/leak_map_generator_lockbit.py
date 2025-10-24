import re
import folium
import requests
import pandas as pd
from collections import defaultdict, Counter
from country_mapping import TLD_TO_COUNTRY, COUNTRY_COORDINATES
from ollama import chat
from pydantic import BaseModel
from datetime import datetime

class CountryIdentification(BaseModel):
    country: str

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running and accessible")
            return True
        else:
            print(f"✗ Ollama returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to Ollama - is it running?")
        print("  Please start Ollama with: ollama serve")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False

def query_ollama_for_country(domain, description):
    """Send domain and description to Ollama to identify the country"""
    try:
        response = chat(
            messages=[
                {
                    'role': 'user',
                    'content': f"""Identify the country for this domain and its description.

Rules:
- Return only the country name in English
- If you cannot determine the country, return "unknown"
- Look for company names, locations, addresses, or other geographical indicators
- Consider domain TLD as a hint but prioritize description content

Domain: {domain}
Description: {description}

What country does this belong to?"""
                }
            ],
            model='granite3.1-dense:2b',
            format=CountryIdentification.model_json_schema(),
        )
        
        # Parse the response using Pydantic
        country_info = CountryIdentification.model_validate_json(response.message.content)
        return country_info.country.strip()
        
    except Exception as e:
        print(f"    - Ollama query error for {domain}: {e}")
        return "unknown"

def parse_leak_data(file_path):
    """Parse the data.txt file and extract domain information"""
    leaks = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by domain entries (each entry starts with a domain name)
    entries = re.split(r'\n(?=[a-zA-Z0-9-]+\.[a-zA-Z]+\n)', content)
    
    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) >= 2:
            domain = lines[0].strip()
            status = lines[1].strip() if len(lines) > 1 else 'unknown'
            
            # Get full description (everything after domain and status)
            description_lines = []
            for i in range(2, len(lines)):
                line = lines[i].strip()
                if line and not line.startswith('Updated:'):
                    description_lines.append(line)
            
            description = ' '.join(description_lines) if description_lines else 'No description available'
            
            # Extract TLD from domain for fallback
            tld_match = re.search(r'\.([a-zA-Z]{2,})$', domain)
            tld = tld_match.group(1).lower() if tld_match else 'unknown'
            
            leaks.append({
                'domain': domain,
                'tld': tld,
                'status': status,
                'description': description
            })
    
    return leaks

def process_leaks_with_ollama(leaks, batch_size=5, save_interval=10):
    """Process leaks using Ollama to identify countries"""
    print(f"Processing {len(leaks)} domains with Ollama in batches of {batch_size}")
    
    results = []
    processed_count = 0
    
    # Load existing results if available
    results_file = 'leak_results.csv'
    try:
        existing_df = pd.read_csv(results_file)
        processed_domains = set(existing_df['domain'].tolist())
        results = existing_df.to_dict('records')
        print(f"Loaded {len(results)} existing results from {results_file}")
    except FileNotFoundError:
        processed_domains = set()
        print("No existing results file found, starting fresh")
    
    batch = []
    for i, leak in enumerate(leaks):
        # Skip if already processed
        if leak['domain'] in processed_domains:
            continue
            
        batch.append(leak)
        
        # Process batch when it reaches batch_size or is the last item
        if len(batch) >= batch_size or i == len(leaks) - 1:
            print(f"\nProcessing batch {processed_count//batch_size + 1} ({len(batch)} domains):")
            
            for leak_item in batch:
                domain = leak_item['domain']
                description = leak_item['description']
                
                print(f"  - Analyzing {domain}...")
                
                # Query Ollama for country identification
                identified_country = query_ollama_for_country(domain, description)
                
                # Fallback to TLD mapping if Ollama returns unknown
                if identified_country.lower() == 'unknown':
                    tld = leak_item['tld']
                    fallback_country = TLD_TO_COUNTRY.get(tld, f'Unknown ({tld})')
                    print(f"    → Ollama: unknown, TLD fallback: {fallback_country}")
                    final_country = fallback_country
                else:
                    print(f"    → Ollama identified: {identified_country}")
                    final_country = identified_country
                
                # Add to results
                result = {
                    'domain': domain,
                    'tld': leak_item['tld'],
                    'status': leak_item['status'],
                    'description': description,
                    'ollama_country': identified_country,
                    'final_country': final_country,
                    'processed_at': datetime.now().isoformat()
                }
                results.append(result)
                processed_count += 1
            
            # Save results every save_interval batches
            if processed_count % (batch_size * save_interval) == 0 or i == len(leaks) - 1:
                df = pd.DataFrame(results)
                df.to_csv(results_file, index=False)
                print(f"  ✓ Saved {len(results)} results to {results_file}")
                
                # Create timestamped backup
                backup_file = f"leak_results_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(backup_file, index=False)
                print(f"  ✓ Created backup: {backup_file}")
            
            # Clear batch for next iteration
            batch = []
    
    return results

def count_leaks_by_country_from_results(results):
    """Count leaks by country from processed results"""
    country_counts = Counter()
    
    for result in results:
        country = result['final_country']
        country_counts[country] += 1
    
    return country_counts

def get_color_for_count(count, max_count):
    """Return color based on leak count intensity"""
    if count == 0:
        return 'gray'
    elif count <= max_count * 0.1:
        return 'green'
    elif count <= max_count * 0.3:
        return 'yellow'
    elif count <= max_count * 0.6:
        return 'orange'
    else:
        return 'red'

def create_leak_map(country_counts, output_file='leak_map.html'):
    """Create an interactive folium map showing leak counts by country"""
    
    # Create base map
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    max_count = max(country_counts.values()) if country_counts else 1
    
    # Add markers for each country with leaks
    for country, count in country_counts.items():
        if country in COUNTRY_COORDINATES:
            coords = COUNTRY_COORDINATES[country]
            color = get_color_for_count(count, max_count)
            
            # Create popup text
            popup_text = f"""
            <b>{country}</b><br>
            Leaked Domains: {count}<br>
            Severity: {color.title()}
            """
            
            # Add circle marker
            folium.CircleMarker(
                location=[coords['lat'], coords['lng']],
                radius=min(5 + (count / max_count) * 20, 25),
                popup=folium.Popup(popup_text, max_width=200),
                color='black',
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 150px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Data Leak Severity</b></p>
    <p><i class="fa fa-circle" style="color:red"></i> Critical (60%+)</p>
    <p><i class="fa fa-circle" style="color:orange"></i> High (30-60%)</p>
    <p><i class="fa fa-circle" style="color:yellow"></i> Medium (10-30%)</p>
    <p><i class="fa fa-circle" style="color:green"></i> Low (0-10%)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = '''
    <h3 align="center" style="font-size:20px"><b>Global Data Leaks by Country</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save map
    m.save(output_file)
    print(f"Map saved as {output_file}")
    return m

def generate_statistics(country_counts):
    """Generate and print statistics"""
    total_leaks = sum(country_counts.values())
    total_countries = len(country_counts)
    
    print(f"\n=== DATA LEAK STATISTICS ===")
    print(f"Total leaked domains: {total_leaks}")
    print(f"Countries affected: {total_countries}")
    print(f"Average leaks per country: {total_leaks / total_countries:.1f}")
    
    print(f"\n=== TOP 10 MOST AFFECTED COUNTRIES ===")
    for i, (country, count) in enumerate(country_counts.most_common(10), 1):
        percentage = (count / total_leaks) * 100
        print(f"{i:2d}. {country:<25} {count:3d} leaks ({percentage:4.1f}%)")

def main():
    """Main function to run the leak map generator with Ollama integration"""
    print("Starting Data Leak Map Generator with AI Country Identification...")
    
    # Check Ollama connection first
    if not check_ollama_connection():
        print("Please start Ollama first with: ollama serve")
        print("Then make sure granite3.1-dense:2b model is available with: ollama pull granite3.1-dense:2b")
        return
    
    # Parse the leak data
    print("Parsing leak data...")
    leaks = parse_leak_data('data.txt')
    print(f"Found {len(leaks)} leak entries")
    
    # Process first batch of domains with Ollama (limit to 5 for testing)
    print("Processing domains with AI country identification...")
    results = process_leaks_with_ollama(leaks, batch_size=5, save_interval=2)
    
    # Count leaks by country from results
    print("Counting leaks by country...")
    country_counts = count_leaks_by_country_from_results(results)
    
    # Generate statistics
    generate_statistics(country_counts)
    
    # Create and save the dot map
    print("\nGenerating interactive dot map...")
    create_leak_map(country_counts)
    
    # Ask user if they want to create choropleth map
    try:
        create_choropleth = input("\nDo you want to create a choropleth map (colors entire countries)? (y/n): ").lower().startswith('y')
        if create_choropleth:
            print("Generating choropleth map...")
            from Final.country_full import create_country_choropleth_map
            df = pd.DataFrame(results)
            country_counts_clean = count_leaks_by_country_choropleth(df)
            create_country_choropleth_map(country_counts_clean)
            print("✓ Choropleth map created as 'leak_map_countries.html'")
    except KeyboardInterrupt:
        print("\nSkipping choropleth map generation.")
    
    print(f"\nDone! Processed {len(results)} domains.")
    print("Check 'leak_results.csv' for detailed results.")
    print("Open 'leak_map.html' in your browser to view the dot map.")

def count_leaks_by_country_choropleth(df):
    """Count leaks by country for choropleth map with cleaned country names"""
    from Final.country_full import clean_country_name
    
    country_counts = Counter()
    for _, row in df.iterrows():
        country = clean_country_name(row['final_country'])
        country_counts[country] += 1
    
    return country_counts

if __name__ == "__main__":
    main()
