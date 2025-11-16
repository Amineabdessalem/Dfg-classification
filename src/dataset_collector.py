"""
Dataset Collection Module for DFG Classifier
Collects scientific papers from various sources (ArXiv, PubMed, etc.)
"""

import os
import json
import logging
import time
import requests
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
from urllib.parse import quote
import random

logger = logging.getLogger(__name__)


class ArXivCollector:
    """Collect papers from ArXiv API"""
    
    def __init__(self, delay: float = 3.0):
        """
        Initialize ArXiv collector
        
        Args:
            delay: Delay between API requests (seconds)
        """
        self.base_url = "http://export.arxiv.org/api/query"
        self.delay = delay
    
    def search(self, 
               query: str, 
               max_results: int = 100,
               category: Optional[str] = None) -> List[Dict]:
        """
        Search ArXiv for papers
        
        Args:
            query: Search query
            max_results: Maximum number of results
            category: ArXiv category (e.g., 'cs.LG', 'physics.atom-ph')
            
        Returns:
            List of paper metadata dictionaries
        """
        papers = []
        
        # Construct search query
        search_query = query
        if category:
            search_query = f"cat:{category} AND {query}"
        
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            logger.info(f"Searching ArXiv with query: {search_query}")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Extract entries
            namespaces = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', namespaces):
                paper = self._parse_entry(entry, namespaces)
                if paper:
                    papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers")
            
            # Respect API rate limits
            time.sleep(self.delay)
            
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
        
        return papers
    
    def _parse_entry(self, entry, namespaces: Dict) -> Optional[Dict]:
        """Parse ArXiv entry XML"""
        try:
            paper = {
                'source': 'arxiv',
                'id': entry.find('atom:id', namespaces).text.split('/')[-1],
                'title': entry.find('atom:title', namespaces).text.strip(),
                'abstract': entry.find('atom:summary', namespaces).text.strip(),
                'authors': [],
                'categories': [],
                'published': entry.find('atom:published', namespaces).text,
                'updated': entry.find('atom:updated', namespaces).text,
                'pdf_url': None,
                'collected_at': datetime.now().isoformat()
            }
            
            # Extract authors
            for author in entry.findall('atom:author', namespaces):
                name_elem = author.find('atom:name', namespaces)
                if name_elem is not None:
                    paper['authors'].append(name_elem.text)
            
            # Extract categories
            for category in entry.findall('atom:category', namespaces):
                term = category.get('term')
                if term:
                    paper['categories'].append(term)
            
            # Extract PDF link
            for link in entry.findall('atom:link', namespaces):
                if link.get('title') == 'pdf':
                    paper['pdf_url'] = link.get('href')
                    break
            
            return paper
            
        except Exception as e:
            logger.warning(f"Error parsing entry: {e}")
            return None
    
    def download_pdf(self, paper: Dict, output_dir: str) -> Optional[str]:
        """Download PDF for a paper"""
        if not paper.get('pdf_url'):
            logger.warning(f"No PDF URL for paper {paper.get('id')}")
            return None
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename
            paper_id = paper['id'].replace('/', '_').replace(':', '_')
            filename = f"{paper_id}.pdf"
            filepath = os.path.join(output_dir, filename)
            
            # Download PDF
            logger.info(f"Downloading PDF: {filename}")
            response = requests.get(paper['pdf_url'], stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded: {filepath}")
            time.sleep(self.delay)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return None


class PubMedCollector:
    """Collect papers from PubMed API"""
    
    def __init__(self, email: str, delay: float = 0.34):
        """
        Initialize PubMed collector
        
        Args:
            email: Your email for PubMed API
            delay: Delay between requests (NCBI requires 0.34s = max 3 requests/sec)
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.email = email
        self.delay = delay
    
    def search(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Search PubMed for papers
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of paper metadata dictionaries
        """
        papers = []
        
        try:
            # Step 1: Search for PMIDs
            pmids = self._search_pmids(query, max_results)
            
            if not pmids:
                logger.warning("No PMIDs found")
                return papers
            
            # Step 2: Fetch paper details
            papers = self._fetch_details(pmids)
            
            logger.info(f"Collected {len(papers)} papers from PubMed")
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
        
        return papers
    
    def _search_pmids(self, query: str, max_results: int) -> List[str]:
        """Search for PMIDs matching query"""
        search_url = f"{self.base_url}/esearch.fcgi"
        
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'email': self.email
        }
        
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        pmids = data.get('esearchresult', {}).get('idlist', [])
        
        time.sleep(self.delay)
        return pmids
    
    def _fetch_details(self, pmids: List[str]) -> List[Dict]:
        """Fetch paper details for given PMIDs"""
        fetch_url = f"{self.base_url}/efetch.fcgi"
        papers = []
        
        # Fetch in batches of 100
        batch_size = 100
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i+batch_size]
            
            params = {
                'db': 'pubmed',
                'id': ','.join(batch_pmids),
                'retmode': 'xml',
                'email': self.email
            }
            
            response = requests.get(fetch_url, params=params)
            response.raise_for_status()
            
            # Parse XML
            root = ET.fromstring(response.content)
            
            for article in root.findall('.//PubmedArticle'):
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            
            time.sleep(self.delay)
        
        return papers
    
    def _parse_article(self, article) -> Optional[Dict]:
        """Parse PubMed article XML"""
        try:
            # Extract article data
            medline_citation = article.find('.//MedlineCitation')
            article_elem = medline_citation.find('.//Article')
            
            title_elem = article_elem.find('.//ArticleTitle')
            abstract_elem = article_elem.find('.//Abstract/AbstractText')
            
            paper = {
                'source': 'pubmed',
                'id': medline_citation.find('.//PMID').text,
                'title': title_elem.text if title_elem is not None else 'No title',
                'abstract': abstract_elem.text if abstract_elem is not None else 'No abstract',
                'authors': [],
                'journal': '',
                'published': '',
                'collected_at': datetime.now().isoformat()
            }
            
            # Extract authors
            for author in article_elem.findall('.//Author'):
                lastname = author.find('.//LastName')
                forename = author.find('.//ForeName')
                if lastname is not None:
                    name = lastname.text
                    if forename is not None:
                        name = f"{forename.text} {name}"
                    paper['authors'].append(name)
            
            # Extract journal
            journal_elem = article_elem.find('.//Journal/Title')
            if journal_elem is not None:
                paper['journal'] = journal_elem.text
            
            # Extract publication date
            pub_date = article_elem.find('.//Journal/JournalIssue/PubDate')
            if pub_date is not None:
                year = pub_date.find('.//Year')
                month = pub_date.find('.//Month')
                if year is not None:
                    date_str = year.text
                    if month is not None:
                        date_str = f"{month.text} {date_str}"
                    paper['published'] = date_str
            
            return paper
            
        except Exception as e:
            logger.warning(f"Error parsing article: {e}")
            return None


class DFGCategoryMapper:
    """Map scientific papers to DFG categories"""
    
    def __init__(self, dfg_mapping: Dict):
        """
        Initialize DFG category mapper
        
        Args:
            dfg_mapping: DFG mapping dictionary
        """
        self.dfg_mapping = dfg_mapping
        self._create_keyword_mapping()
    
    def _create_keyword_mapping(self):
        """Create keyword-to-category mapping"""
        # This is a simplified mapping - in production, use ML-based approach
        self.keyword_mapping = {
            # Computer Science & Engineering
            'machine learning': ['3.15', '4.13'],
            'artificial intelligence': ['3.15', '4.13'],
            'deep learning': ['3.15'],
            'neural network': ['3.15', '4.13'],
            'computer vision': ['3.15'],
            'robotics': ['4.11', '4.13'],
            'software engineering': ['3.15'],
            
            # Physics
            'quantum': ['3.12'],
            'particle': ['3.12'],
            'condensed matter': ['3.12'],
            'astrophysics': ['3.12'],
            'optics': ['3.12'],
            
            # Chemistry
            'synthesis': ['3.13'],
            'catalysis': ['3.13'],
            'organic chemistry': ['3.13'],
            'materials': ['4.15', '3.13'],
            
            # Biology & Medicine
            'cancer': ['2.12'],
            'genome': ['2.11'],
            'protein': ['2.11'],
            'cell': ['2.11'],
            'medicine': ['2.12'],
            'clinical': ['2.12'],
            
            # Mathematics
            'theorem': ['3.11'],
            'optimization': ['3.11', '3.15'],
            'statistics': ['3.11'],
            
            # Social Sciences
            'psychology': ['2.14'],
            'economics': ['2.17'],
            'sociology': ['2.16'],
            'education': ['2.15'],
            
            # Humanities
            'philosophy': ['1.13'],
            'history': ['1.12'],
            'linguistics': ['1.15'],
        }
    
    def suggest_category(self, paper: Dict) -> Optional[str]:
        """
        Suggest DFG category for a paper
        
        Args:
            paper: Paper metadata
            
        Returns:
            Suggested DFG category code
        """
        # Combine title and abstract
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        
        # Score categories based on keywords
        category_scores = {}
        
        for keyword, categories in self.keyword_mapping.items():
            if keyword in text:
                for cat in categories:
                    category_scores[cat] = category_scores.get(cat, 0) + 1
        
        # Return category with highest score
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        
        return None


def collect_dfg_dataset(
    output_dir: str,
    dfg_mapping: Dict,
    papers_per_category: int = 50,
    sources: List[str] = ['arxiv', 'pubmed']
) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Collect comprehensive DFG dataset
    
    Args:
        output_dir: Output directory for collected data
        dfg_mapping: DFG mapping dictionary
        papers_per_category: Target number of papers per category
        sources: List of sources to collect from
        
    Returns:
        Tuple of (papers list, labels dictionary)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_papers = []
    labels = {}
    
    # Initialize collectors
    arxiv_collector = ArXivCollector() if 'arxiv' in sources else None
    pubmed_collector = PubMedCollector(email="your.email@example.com") if 'pubmed' in sources else None
    
    # Initialize category mapper
    mapper = DFGCategoryMapper(dfg_mapping)
    
    # Get DFG categories
    level_2_classes = dfg_mapping.get('level_2', {}).get('classes', {})
    
    logger.info(f"Collecting papers for {len(level_2_classes)} categories")
    
    # Define search queries for each category
    category_queries = _get_category_queries()
    
    for category_code, category_name in level_2_classes.items():
        logger.info(f"Collecting papers for {category_code}: {category_name}")
        
        queries = category_queries.get(category_code, [category_name.lower()])
        
        category_papers = []
        
        for query in queries:
            if arxiv_collector and len(category_papers) < papers_per_category:
                papers = arxiv_collector.search(query, max_results=papers_per_category // len(queries))
                category_papers.extend(papers)
            
            if pubmed_collector and len(category_papers) < papers_per_category:
                papers = pubmed_collector.search(query, max_results=papers_per_category // len(queries))
                category_papers.extend(papers)
            
            if len(category_papers) >= papers_per_category:
                break
        
        # Label collected papers
        for paper in category_papers[:papers_per_category]:
            paper_id = f"{paper['source']}_{paper['id']}"
            labels[paper_id] = category_code
            all_papers.append(paper)
    
    # Save collected data
    papers_file = os.path.join(output_dir, 'collected_papers.json')
    labels_file = os.path.join(output_dir, 'collected_labels.json')
    
    with open(papers_file, 'w', encoding='utf-8') as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)
    
    with open(labels_file, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Collected {len(all_papers)} papers with labels")
    logger.info(f"Saved to {papers_file} and {labels_file}")
    
    return all_papers, labels


def _get_category_queries() -> Dict[str, List[str]]:
    """Get search queries for each DFG category"""
    return {
        # Natural Sciences
        '3.11': ['mathematics', 'algebra', 'topology', 'number theory'],
        '3.12': ['physics', 'quantum mechanics', 'thermodynamics'],
        '3.13': ['chemistry', 'organic chemistry', 'catalysis'],
        '3.14': ['geosciences', 'geology', 'climate'],
        '3.15': ['computer science', 'machine learning', 'algorithms'],
        '3.16': ['systems engineering', 'control systems'],
        
        # Life Sciences
        '2.11': ['molecular biology', 'genetics', 'cell biology'],
        '2.12': ['medicine', 'clinical research', 'pathology'],
        '2.13': ['agriculture', 'forestry', 'veterinary'],
        '2.14': ['psychology', 'cognitive science'],
        '2.15': ['education research', 'pedagogy'],
        '2.16': ['sociology', 'social sciences'],
        '2.17': ['economics', 'econometrics'],
        '2.18': ['law', 'jurisprudence'],
        
        # Engineering
        '4.11': ['mechanical engineering', 'manufacturing'],
        '4.12': ['civil engineering', 'architecture'],
        '4.13': ['electrical engineering', 'signal processing'],
        '4.14': ['production technology'],
        '4.15': ['materials science', 'metallurgy'],
        '4.16': ['chemical engineering', 'process engineering'],
        
        # Humanities
        '1.11': ['archaeology', 'ancient cultures'],
        '1.12': ['history', 'historical research'],
        '1.13': ['philosophy', 'ethics'],
        '1.14': ['theology', 'religious studies'],
        '1.15': ['linguistics', 'literature'],
        '1.16': ['art history', 'musicology'],
    }


if __name__ == "__main__":
    # Test data collection
    logging.basicConfig(level=logging.INFO)
    
    # Test ArXiv collector
    arxiv = ArXivCollector()
    papers = arxiv.search("machine learning", max_results=5)
    
    print(f"Found {len(papers)} papers from ArXiv")
    if papers:
        print(f"First paper: {papers[0]['title']}")




