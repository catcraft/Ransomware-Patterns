import csv
import os
from datetime import datetime
from urllib.parse import urlparse
from Utils.country_mapping import TLD_TO_COUNTRY

def extract_domain_info(url):
    """Extract domain and TLD from URL"""
    # Remove protocol if present
    if '://' in url:
        url = url.split('://', 1)[1]
    
    # Remove path if present
    if '/' in url:
        url = url.split('/', 1)[0]
    
    # Extract TLD
    parts = url.split('.')
    if len(parts) >= 2:
        domain = url
        tld = parts[-1]
        return domain, tld
    else:
        return url, ''

def get_country_from_tld(tld):
    """Get country from TLD using the mapping"""
    return TLD_TO_COUNTRY.get(tld.lower(), 'Unknown')

def process_clop_file():
    """Process clop.txt and generate results.csv"""
    input_file = 'clop.txt'
    output_file = 'results.csv'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
        return
    
    results = []
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Read URLs from clop.txt
    with open(input_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]
    
    # Process each URL
    for url in urls:
        domain, tld = extract_domain_info(url)
        country = get_country_from_tld(tld)
        
        result = {
            'domain': domain,
            'tld': tld,
            'status': 'published',
            'description': '-',
            'ollama_country': country,
            'final_country': country,
            'processed_at': current_time
        }
        results.append(result)
    
    # Write results to CSV
    fieldnames = ['domain', 'tld', 'status', 'description', 'ollama_country', 'final_country', 'processed_at']
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Processed {len(results)} URLs and saved to {output_file}")

if __name__ == "__main__":
    process_clop_file()
