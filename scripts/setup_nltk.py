import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def setup_nltk():
    """Download required NLTK data"""
    required_packages = [
        'punkt',
        'wordnet',
        'omw-1.4'
    ]
    
    for package in required_packages:
        try:
            nltk.download(package, quiet=True)
            print(f"Successfully downloaded {package}")
        except Exception as e:
            print(f"Error downloading {package}: {str(e)}")

if __name__ == "__main__":
    setup_nltk() 