# setup.py - Updated with folder creation

import subprocess
import sys
import os

def install_requirements():
    """Install all required packages and create directories."""
    print("=" * 60)
    print("EMOTIONAL TTS PROJECT SETUP")
    print("=" * 60)
    
    print("\n📦 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✅ All packages installed successfully!")
    except Exception as e:
        print(f"\n❌ Error installing packages: {e}")
        return False
    
    print("\n📁 Creating necessary directories...")
    directories = [
        'checkpoints', 
        'audio_outputs', 
        'logs',
        'uploads',
        'src/web_interface/static',
        'src/web_interface/templates'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ Created: {directory}/")
    
    print("\n" + "=" * 60)
    print("✅ SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run: jupyter notebook")
    print("2. Open: notebooks/emotional_tts_complete.ipynb")
    print("3. Or run: python run_web_app.py")
    print("\n" + "=" * 60)
    
    return True

if __name__ == "__main__":
    install_requirements()
