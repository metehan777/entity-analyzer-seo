import os
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from google.cloud import aiplatform
from google.cloud import language_v1
from typing import Dict, List, Set, Tuple

# Hardcoded credentials
GOOGLE_API_KEY = "XXXXXX"  # Using client_secret as API key
GOOGLE_CLOUD_PROJECT = "metehan777"
GOOGLE_APPLICATION_CREDENTIALS = "metehan777-6ff68b6435e0.json"
LOCATION = "us-central1"

class KnowledgeGapAnalyzer:
    def __init__(self):
        """
        Initialize the analyzer with hardcoded credentials.
        """
        self.google_api_key = GOOGLE_API_KEY
        self.project_id = GOOGLE_CLOUD_PROJECT
        self.location = LOCATION
        self.knowledge_graph_endpoint = "https://kgsearch.googleapis.com/v1/entities:search"
        
        # Set the environment variable for service account
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
        
        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=self.location)
        self.language_client = language_v1.LanguageServiceClient()

    def extract_content(self, url: str) -> str:
        """Extract and clean text content from a given URL."""
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean and normalize text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            raise Exception(f"Error extracting content from URL: {str(e)}")

    def query_knowledge_graph(self, entity: str) -> Dict:
        """Query Google Knowledge Graph API for entity information."""
        params = {
            'query': entity,
            'key': self.google_api_key,
            'limit': 1
        }
        
        response = requests.get(self.knowledge_graph_endpoint, params=params)
        return response.json()

    def analyze_with_vertex(self, text: str) -> Dict:
        """Analyze text content using Vertex AI Text Analysis."""
        document = language_v1.Document(
            content=text,
            type_=language_v1.Document.Type.PLAIN_TEXT
        )
        
        # Analyze entities
        entities = self.language_client.analyze_entities(
            request={'document': document}
        )
        
        # Analyze entity sentiment
        entity_sentiment = self.language_client.analyze_entity_sentiment(
            request={'document': document}
        )
        
        return {
            'entities': entities,
            'entity_sentiment': entity_sentiment
        }

    def analyze_content(self, url: str) -> Dict:
        """
        Analyze URL content and identify entities with detailed metadata.
        """
        # Extract content
        content = self.extract_content(url)
        
        # Analyze with Vertex AI
        vertex_analysis = self.analyze_with_vertex(content)
        
        # Process entities and their metadata
        entities_data = []
        found_entities = set()
        
        for entity in vertex_analysis['entities'].entities:
            entity_info = {
                'name': entity.name,
                'type': language_v1.Entity.Type(entity.type_).name,
                'salience': entity.salience,
                'metadata': dict(entity.metadata),
                'mentions': len(entity.mentions),
                'sentiment': {
                    'magnitude': vertex_analysis['entity_sentiment'].entities[0].sentiment.magnitude,
                    'score': vertex_analysis['entity_sentiment'].entities[0].sentiment.score
                } if vertex_analysis['entity_sentiment'].entities else None
            }
            
            # Query Knowledge Graph for additional information
            kg_result = self.query_knowledge_graph(entity.name)
            if 'itemListElement' in kg_result and kg_result['itemListElement']:
                item = kg_result['itemListElement'][0]
                if 'result' in item:
                    entity_info['knowledge_graph'] = item['result']
            
            entities_data.append(entity_info)
            found_entities.add(entity.name)

        return {
            'url': url,
            'entities_data': entities_data,
            'entity_count': len(found_entities)
        }

def main():
    # Example usage
    url = input("Enter the URL to analyze: ")  # Get URL from user input
    
    # Initialize analyzer
    analyzer = KnowledgeGapAnalyzer()
    
    print(f"\nAnalyzing {url}...")
    
    # Analyze URL
    results = analyzer.analyze_content(url)
    
    # Save detailed results to a text file
    with open('entities_analysis.txt', 'w', encoding='utf-8') as f:
        f.write(f"Analysis Results for {results['url']}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Entities Found: {results['entity_count']}\n\n")
        
        f.write("Detailed Entity Analysis:\n")
        f.write("=" * 50 + "\n")
        
        # Sort entities by salience score
        sorted_entities = sorted(results['entities_data'], 
                               key=lambda x: x['salience'], 
                               reverse=True)
        
        for entity in sorted_entities:
            f.write(f"\nEntity: {entity['name']}\n")
            f.write(f"Type: {entity['type']}\n")
            f.write(f"Salience Score: {entity['salience']:.4f}\n")
            f.write(f"Number of Mentions: {entity['mentions']}\n")
            
            if entity['sentiment']:
                f.write(f"Sentiment Score: {entity['sentiment']['score']:.2f}\n")
                f.write(f"Sentiment Magnitude: {entity['sentiment']['magnitude']:.2f}\n")
            
            if entity['metadata']:
                f.write("Metadata:\n")
                for key, value in entity['metadata'].items():
                    f.write(f"  {key}: {value}\n")
            
            if 'knowledge_graph' in entity:
                f.write("Knowledge Graph Information:\n")
                kg_info = entity['knowledge_graph']
                if 'description' in kg_info:
                    f.write(f"  Description: {kg_info['description']}\n")
                if '@type' in kg_info:
                    f.write(f"  Type: {kg_info['@type']}\n")
                if 'detailedDescription' in kg_info:
                    f.write(f"  Detailed Description: {kg_info['detailedDescription']['articleBody']}\n")
            
            f.write("-" * 40 + "\n")
    
    print(f"\nAnalysis complete! Results saved to 'entities_analysis.txt'")
    print(f"Found {results['entity_count']} entities")

if __name__ == "__main__":
    main()