"""
GEPRIS Project Scraper for DFG Classification
Scrapes Computer Science (443) and Electrical Engineering (442) projects
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GEPRISScraper:
    """Scraper for GEPRIS DFG project database"""
    
    BASE_URL = "https://gepris.dfg.de/gepris/OCTOPUS"
    
    # Fachkollegium to Level-2 DFG code mapping
    FACHKOLLEGIUM_MAP = {
        "441": "4.41",  # Systems Engineering
        "442": "4.42",  # Electrical Engineering
        "443": "4.43",  # Computer Science
    }
    
    def __init__(self, delay: float = 2.0):
        """
        Initialize scraper
        
        Args:
            delay: Delay between requests in seconds (be polite!)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_project_list(
        self, 
        fachkollegium: str, 
        max_projects: int = 100
    ) -> List[str]:
        """
        Scrape project URLs from listing pages
        
        Args:
            fachkollegium: Fachkollegium code (e.g., "443" for Computer Science)
            max_projects: Maximum number of project URLs to collect
            
        Returns:
            List of project detail page URLs
        """
        project_urls = []
        index = 0
        hits_per_page = 50
        
        logger.info(f"Scraping Fachkollegium {fachkollegium} project list...")
        
        while len(project_urls) < max_projects:
            # Build listing page URL
            params = {
                'context': 'projekt',
                'fachgebiet': '44',  # Computer Science, Systems and Electrical Engineering
                'fachkollegium': fachkollegium,
                'task': 'doKatalog',
                'index': str(index),
                'hitsPerPage': str(hits_per_page),
                'nurProjekteMitAB': 'false',
                'teilprojekte': 'true',
                'zk_transferprojekt': 'false'
            }
            
            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find project links (they typically have projekt context)
                links = soup.find_all('a', href=True)
                page_projects = []
                
                for link in links:
                    href = link.get('href', '')
                    if 'context=projekt' in href and 'id=' in href:
                        full_url = urljoin(self.BASE_URL, href)
                        if full_url not in project_urls and full_url not in page_projects:
                            page_projects.append(full_url)
                
                if not page_projects:
                    logger.info(f"No more projects found at index {index}")
                    break
                
                project_urls.extend(page_projects)
                logger.info(f"Found {len(page_projects)} projects at index {index} (total: {len(project_urls)})")
                
                # Move to next page
                index += hits_per_page
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Error fetching listing page at index {index}: {e}")
                break
        
        return project_urls[:max_projects]
    
    def scrape_project_details(self, project_url: str) -> Optional[Dict]:
        """
        Scrape details from a single project page
        
        Args:
            project_url: URL of project detail page
            
        Returns:
            Dictionary with title, abstract, and dfg_label, or None if failed
        """
        try:
            response = self.session.get(project_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title (usually in h1 or h2)
            title = None
            for tag in ['h1', 'h2']:
                title_elem = soup.find(tag)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break
            
            if not title:
                logger.warning(f"No title found for {project_url}")
                return None
            
            # Extract abstract from content div
            abstract = None
            content_div = soup.find('div', class_='content')
            if content_div:
                # Get all text from paragraphs
                paragraphs = content_div.find_all('p')
                if paragraphs:
                    abstract = ' '.join(p.get_text(strip=True) for p in paragraphs)
                else:
                    abstract = content_div.get_text(strip=True)
            
            if not abstract or len(abstract) < 50:
                logger.warning(f"No substantial abstract found for {project_url}")
                return None
            
            # Extract Fachkollegium code
            fachkollegium = None
            page_text = soup.get_text()
            
            # Look for "Fachkollegium = XXX" pattern
            import re
            fk_match = re.search(r'Fachkollegium\s*[=:]\s*(\d{3})', page_text)
            if fk_match:
                fachkollegium = fk_match.group(1)
            
            if not fachkollegium or fachkollegium not in self.FACHKOLLEGIUM_MAP:
                logger.warning(f"No valid Fachkollegium found for {project_url}")
                return None
            
            # Map to DFG Level-2 code
            dfg_label = self.FACHKOLLEGIUM_MAP[fachkollegium]
            
            # Check if text is in German and needs translation
            # Simple heuristic: check for common German words
            german_indicators = ['der', 'die', 'das', 'und', 'oder', 'für', 'mit', 'von', 'zu']
            text_sample = (title + ' ' + abstract[:200]).lower()
            is_german = sum(1 for word in german_indicators if f' {word} ' in text_sample) >= 2
            
            if is_german:
                logger.info(f"Detected German text, translating...")
                title = self._translate_text(title)
                abstract = self._translate_text(abstract)
                
                if not title or not abstract:
                    logger.warning(f"Translation failed for {project_url}")
                    return None
            
            return {
                'title': title,
                'abstract': abstract,
                'dfg_label': dfg_label,
                'source_url': project_url
            }
            
        except Exception as e:
            logger.error(f"Error scraping {project_url}: {e}")
            return None
    
    def _translate_text(self, text: str) -> Optional[str]:
        """
        Translate German text to English using a free translation service
        
        Args:
            text: German text to translate
            
        Returns:
            Translated English text or None if failed
        """
        try:
            # Use MyMemory Translation API (free, no key required)
            # Limit: 500 chars per request, 1000 requests/day
            if len(text) > 500:
                # Split into chunks
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                translated_chunks = []
                for chunk in chunks:
                    translated = self._translate_chunk(chunk)
                    if translated:
                        translated_chunks.append(translated)
                    time.sleep(0.5)  # Extra delay for multiple requests
                return ' '.join(translated_chunks) if translated_chunks else None
            else:
                return self._translate_chunk(text)
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return None
    
    def _translate_chunk(self, text: str) -> Optional[str]:
        """Translate a single chunk of text"""
        try:
            url = "https://api.mymemory.translated.net/get"
            params = {
                'q': text,
                'langpair': 'de|en'
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('responseStatus') == 200:
                return data['responseData']['translatedText']
            else:
                logger.warning(f"Translation API returned status: {data.get('responseStatus')}")
                return None
                
        except Exception as e:
            logger.error(f"Translation chunk error: {e}")
            return None
    
    def scrape_fachkollegium(
        self, 
        fachkollegium: str, 
        max_projects: int = 100,
        output_file: Optional[str] = None
    ) -> List[Dict]:
        """
        Scrape all projects for a given Fachkollegium
        
        Args:
            fachkollegium: Fachkollegium code (e.g., "443")
            max_projects: Maximum number of projects to scrape
            output_file: Optional file to save results
            
        Returns:
            List of project dictionaries
        """
        logger.info(f"Starting scrape for Fachkollegium {fachkollegium}")
        
        # Get project URLs
        project_urls = self.scrape_project_list(fachkollegium, max_projects)
        logger.info(f"Found {len(project_urls)} project URLs")
        
        # Scrape each project
        projects = []
        for i, url in enumerate(project_urls, 1):
            logger.info(f"Scraping project {i}/{len(project_urls)}: {url}")
            
            project = self.scrape_project_details(url)
            if project:
                projects.append(project)
                logger.info(f"✓ Successfully scraped project {i} (total: {len(projects)})")
            else:
                logger.warning(f"✗ Failed to scrape project {i}")
            
            time.sleep(self.delay)
        
        logger.info(f"Scraped {len(projects)} valid projects out of {len(project_urls)} URLs")
        
        # Save if output file specified
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(projects, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved to {output_file}")
        
        return projects


def main():
    """Main scraping function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape GEPRIS DFG projects")
    parser.add_argument(
        '--fachkollegium',
        type=str,
        required=True,
        choices=['441', '442', '443'],
        help='Fachkollegium code to scrape'
    )
    parser.add_argument(
        '--max-projects',
        type=int,
        default=100,
        help='Maximum number of projects to scrape'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/gepris/scraped_projects.json',
        help='Output JSON file'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=2.0,
        help='Delay between requests in seconds'
    )
    
    args = parser.parse_args()
    
    scraper = GEPRISScraper(delay=args.delay)
    projects = scraper.scrape_fachkollegium(
        fachkollegium=args.fachkollegium,
        max_projects=args.max_projects,
        output_file=args.output
    )
    
    print(f"\n{'='*60}")
    print(f"SCRAPING COMPLETE")
    print(f"{'='*60}")
    print(f"Fachkollegium: {args.fachkollegium}")
    print(f"Projects scraped: {len(projects)}")
    print(f"Output file: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

