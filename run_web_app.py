# run_web_app.py - Launch Flask web application

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from web_interface.app import app

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 STARTING EMOTIONAL TTS WEB APPLICATION")
    print("=" * 60)
    print("\n✅ Server starting...")
    print("🌐 Open your browser at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
