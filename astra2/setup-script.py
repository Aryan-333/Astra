import subprocess
import sys

def install_requirements():
    """Install the required packages for the jewelry editor application"""
    requirements = [
        "streamlit",
        "torch",
        "torchvision",
        "ftfy",
        "regex",
        "tqdm",
        "open_clip_torch",
        "spacy",
        "replicate",
        "opencv-python",
        "matplotlib",
        "Pillow",
        "requests"
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Download spaCy model
    print("Downloading spaCy model...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    
    print("All dependencies installed successfully!")

if __name__ == "__main__":
    install_requirements()