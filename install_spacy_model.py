import subprocess
import sys

def download_model():
    try:
        subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_md'])
    except subprocess.CalledProcessError as e:
        print(f"Failed to download the en_core_web_md model: {e}")

if __name__ == '__main__':
    download_model()
