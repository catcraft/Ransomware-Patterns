import re
import folium
import requests
import pandas as pd
from collections import defaultdict, Counter
from Utils.country_mapping import TLD_TO_COUNTRY, COUNTRY_COORDINATES
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

def query_ollama_for_country(domain, description, company_name=None):
    """Send domain and description to Ollama to identify the country"""
    try:
        # Use company name if available for better context
        context = f"Company: {company_name}\n" if company_name else ""
        context += f"Domain: {domain}\nDescription: {description}"
        
        response = chat(
            messages=[
                {
                    'role': 'user',
                    'content': f"""Identify the country for this company and its description.

Rules:
- Return only the country name in English
- If you cannot determine the country, return "unknown"
- Look for company names, locations, addresses, or other geographical indicators
- Consider any geographical clues in the company name or description

{context}

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
    """Parse the DragonForce data.txt file and extract company information"""
    leaks = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by entries - each entry starts with a date
    entries = re.split(r'\n(?=\d{4}-\d{2}-\d{2})', content.strip())
    
    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) >= 2:
            # First line is the date
            date = lines[0].strip()
            
            # Second line contains tab-separated data: company name and description
            if len(lines) > 1:
                # Split by tabs and clean up
                parts = lines[1].split('\t')
                # Remove empty parts and strip whitespace
                parts = [part.strip() for part in parts if part.strip()]
                
                if len(parts) >= 2:
                    # First non-empty part is company name, second is description
                    company_name = parts[0]
                    description = parts[1] if len(parts) > 1 else ''
                elif len(parts) == 1:
                    # Only company name available
                    company_name = parts[0]
                    description = ''
                else:
                    # Skip entries without proper data
                    continue
            else:
                continue
            
            # If description is empty or very short, try to get it from subsequent lines
            if not description or len(description) < 20:
                description_lines = []
                for i in range(2, len(lines)):
                    line = lines[i].strip()
                    if line and not line.startswith('Updated:') and not line.startswith('Screen') and line != 'Screen':
                        description_lines.append(line)
                if description_lines:
                    additional_desc = ' '.join(description_lines)
                    description = description + ' ' + additional_desc if description else additional_desc
            
            # Skip if we don't have a valid company name
            if not company_name:
                continue
            
            # Create a pseudo-domain from company name for consistency
            # Remove special characters and spaces, convert to lowercase
            domain_base = re.sub(r'[^a-zA-Z0-9\s]', '', company_name).replace(' ', '').lower()
            if domain_base:
                domain = domain_base + '.com'  # Add .com for consistency
            else:
                domain = 'unknown.com'
            
            # Extract potential TLD or set as unknown
            tld = 'com'  # Default since we're creating pseudo-domains
            
            leaks.append({
                'domain': domain,
                'company_name': company_name,
                'tld': tld,
                'status': 'breached',  # All DragonForce entries are breaches
                'description': description,
                'date': date
            })
    
    return leaks

def process_leaks_with_ollama(leaks, batch_size=5, save_interval=10):
    """Process leaks using Ollama to identify countries"""
    print(f"Processing {len(leaks)} companies with Ollama in batches of {batch_size}")
    
    results = []
    processed_count = 0
    
    # Load existing results if available
    results_file = 'dragonforce_leak_results.csv'
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
            print(f"\nProcessing batch {processed_count//batch_size + 1} ({len(batch)} companies):")
            
            for leak_item in batch:
                domain = leak_item['domain']
                company_name = leak_item['company_name']
                description = leak_item['description']
                
                print(f"  - Analyzing {company_name} ({domain})...")
                
                # Query Ollama for country identification
                identified_country = query_ollama_for_country(domain, description, company_name)
                
                # No TLD fallback for DragonForce data since domains are generated
                if identified_country.lower() == 'unknown':
                    print(f"    → Ollama: unknown, no fallback available")
                    final_country = "Unknown"
                else:
                    print(f"    → Ollama identified: {identified_country}")
                    final_country = identified_country
                
                # Add to results
                result = {
                    'domain': domain,
                    'company_name': company_name,
                    'tld': leak_item['tld'],
                    'status': leak_item['status'],
                    'description': description,
                    'date': leak_item['date'],
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
                backup_file = f"dragonforce_results_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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

def create_leak_map(country_counts, output_file='dragonforce_leak_map.html'):
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
            Breached Companies: {count}<br>
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
    <p><b>DragonForce Breach Severity</b></p>
    <p><i class="fa fa-circle" style="color:red"></i> Critical (60%+)</p>
    <p><i class="fa fa-circle" style="color:orange"></i> High (30-60%)</p>
    <p><i class="fa fa-circle" style="color:yellow"></i> Medium (10-30%)</p>
    <p><i class="fa fa-circle" style="color:green"></i> Low (0-10%)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = '''
    <h3 align="center" style="font-size:20px"><b>DragonForce Breaches by Country</b></h3>
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
    
    print(f"\n=== DRAGONFORCE BREACH STATISTICS ===")
    print(f"Total breached companies: {total_leaks}")
    print(f"Countries affected: {total_countries}")
    print(f"Average breaches per country: {total_leaks / total_countries:.1f}")
    
    print(f"\n=== TOP 10 MOST AFFECTED COUNTRIES ===")
    for i, (country, count) in enumerate(country_counts.most_common(10), 1):
        percentage = (count / total_leaks) * 100
        print(f"{i:2d}. {country:<25} {count:3d} companies ({percentage:4.1f}%)")

def main():
    """Main function to run the leak map generator with Ollama integration"""
    print("Starting DragonForce Breach Map Generator with AI Country Identification...")
    
    # Check Ollama connection first
    if not check_ollama_connection():
        print("Please start Ollama first with: ollama serve")
        print("Then make sure granite3.1-dense:2b model is available with: ollama pull granite3.1-dense:2b")
        return
    
    # Parse the leak data
    print("Parsing DragonForce breach data...")
    leaks = parse_leak_data('data_dragonforce.txt')
    print(f"Found {len(leaks)} breach entries")
    
    # Process domains with Ollama
    print("Processing companies with AI country identification...")
    results = process_leaks_with_ollama(leaks, batch_size=5, save_interval=2)
    
    # Count leaks by country from results
    print("Counting breaches by country...")
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
            from Generators.Merged_Data.country_full_merged import create_country_choropleth_map
            df = pd.DataFrame(results)
            country_counts_clean = count_leaks_by_country_choropleth(df)
            create_country_choropleth_map(country_counts_clean)
            print("✓ Choropleth map created as 'dragonforce_leak_map_countries.html'")
    except KeyboardInterrupt:
        print("\nSkipping choropleth map generation.")
    
    print(f"\nDone! Processed {len(results)} companies.")
    print("Check 'dragonforce_leak_results.csv' for detailed results.")
    print("Open 'dragonforce_leak_map.html' in your browser to view the dot map.")

def count_leaks_by_country_choropleth(df):
    """Count leaks by country for choropleth map with cleaned country names"""
    from Generators.Merged_Data.country_full_merged import clean_country_name
    
    country_counts = Counter()
    for _, row in df.iterrows():
        country = clean_country_name(row['final_country'])
        country_counts[country] += 1
    
    return country_counts

if __name__ == "__main__":
    main()
