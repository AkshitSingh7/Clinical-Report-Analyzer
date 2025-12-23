import os
import sys
import gdown

def setup_environment():
    print("📦 Setting up environment...")
    
    # 1. Clone the NegBio/ClinicalReport repository
    if not os.path.exists('ClinicalReport'):
        print("\n📥 Cloning repository from GitHub...")
        os.system('git clone https://github.com/ayushnangia/ClinicalReport.git')
        print("✓ Repository cloned successfully!")
    else:
        print("\n✓ Repository already exists")

    # 2. Download BERT Model
    model_path = 'models/tf_model.h5'
    if not os.path.exists(model_path):
        print("\n📥 Downloading BERT model from Google Drive...")
        print("This is a large file (~1.3GB)...")
        os.makedirs('models', exist_ok=True)
        
        file_id = '1FzQteAgbYFHAwPO7Xtd51AJ-hL94X5Xe'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
        print("\n✓ Model downloaded successfully!")
    else:
        print("\n✓ Model already exists")

    # 3. Download NLTK data
    import nltk
    print("\n📥 Downloading NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("✓ NLTK data downloaded!")

if __name__ == "__main__":
    setup_environment()
