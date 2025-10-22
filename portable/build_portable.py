#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portable Builder - Build standalone executable directly from Minimal
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def build_portable():
    """Build portable executable directly from Minimal codebase"""
    
    print("=== Portable Builder ===")
    
    # Get current directory (should be in portable/)
    portable_dir = Path(__file__).parent
    minimal_dir = portable_dir.parent
    
    print(f"Portable directory: {portable_dir}")
    print(f"Minimal directory: {minimal_dir}")
    
    # Check if we're in the right location
    if not (minimal_dir / "transcriber.py").exists():
        print("Error: Cannot find Minimal codebase. Make sure this script is in Minimal/portable/")
        return False
    
    # Create temporary build directory (will be cleaned up)
    build_dir = portable_dir / "temp_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()
    
    try:
        # Copy only necessary files to temp build directory
        print("Preparing build files...")
        
        # Core files to copy
        files_to_copy = [
            "api_transcriber.py",
            "config_utils.py", 
            "recorder.py",
            "transcriber.py",
            "text_cleaner.py",
            "keyboard_typer.py",
            "unified_vad.py",
            "requirements.txt"
        ]
        
        for file_name in files_to_copy:
            src_file = minimal_dir / file_name
            if src_file.exists():
                shutil.copy2(src_file, build_dir / file_name)
                print(f"  - Copied {file_name}")
            else:
                print(f"  - Warning: {file_name} not found")
        
        # Copy realtime directory
        realtime_src = minimal_dir / "realtime"
        realtime_dst = build_dir / "realtime"
        if realtime_src.exists():
            shutil.copytree(realtime_src, realtime_dst)
            print(f"  - Copied realtime/ directory")
        
        # Create portable main script
        create_portable_main(build_dir)
        
        # Build executable from temp directory
        print("Building standalone executable...")
        success = build_executable(build_dir, portable_dir)
        
        if success:
            print("Build successful!")
            print(f"Executable location: {portable_dir / 'dist' / 'WhisperPortable.exe'}")
            
            # Clean up temp build directory
            cleanup_temp_files(portable_dir)
            
            return True
        else:
            print("Build failed!")
            return False
            
    except Exception as e:
        print(f"Error during build: {e}")
        return False

def create_portable_main(build_dir):
    """Create the main portable script"""
    
    main_script = build_dir / "whisper_portable.py"
    
    script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper Portable - Standalone executable version
"""

import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def get_api_key():
    """Get API key from environment variable or user input"""
    # First try environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        print("Using API key from environment variable")
        return api_key

    # If not found, ask user
    print("OpenAI API key not found in environment variables")
    print("Please enter your OpenAI API key:")

    while True:
        api_key = input("API Key: ").strip()

        if api_key:
            print("API key saved for this session")
            return api_key
        else:
            print("Please enter a valid API key")

def main():
    """Main entry point - portable version"""
    print("=== Whisper Portable ===")
    print("Standalone portable version")
    print("Press Ctrl+C to exit")
    print()

    try:
        # Get API key
        api_key = get_api_key()

        # Set environment variable for the app
        os.environ["OPENAI_API_KEY"] = api_key

        # Import and run the main app
        from api_transcriber import App
        app = App()
        app.start()
    except KeyboardInterrupt:
        print("\\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open(main_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"  - Created portable main script")

def build_executable(build_dir, portable_dir):
    """Build standalone executable using PyInstaller"""
    
    try:
        # Change to build directory
        original_cwd = os.getcwd()
        os.chdir(build_dir)
        
        # PyInstaller command - create completely standalone executable
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",                    # Single executable file
            "--console",                    # Show console window
            "--name", "WhisperPortable",    # Executable name
            "--distpath", str(portable_dir / "dist"),  # Output directory
            "--workpath", str(portable_dir / "build"), # Work directory
            "--specpath", str(portable_dir),           # Spec file location
            "--clean",                      # Clean cache
            "--noconfirm",                  # Overwrite without asking
            "whisper_portable.py"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Run PyInstaller
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Change back to original directory
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            print("PyInstaller completed successfully")
            return True
        else:
            print("PyInstaller failed with return code", result.returncode)
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"Error running PyInstaller: {e}")
        return False

def cleanup_temp_files(portable_dir):
    """Clean up temporary build files"""
    print("Cleaning up temporary files...")
    
    try:
        # Remove temp build directory
        temp_build = portable_dir / "temp_build"
        if temp_build.exists():
            shutil.rmtree(temp_build)
            print("  - Removed temp_build/ directory")
        
        # Remove build directory
        build_dir = portable_dir / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
            print("  - Removed build/ directory")
        
        # Remove spec file
        spec_file = portable_dir / "WhisperPortable.spec"
        if spec_file.exists():
            spec_file.unlink()
            print("  - Removed WhisperPortable.spec")
        
        print("Temporary files cleanup completed!")
        
    except Exception as e:
        print(f"Warning: Could not clean up some files: {e}")

if __name__ == "__main__":
    success = build_portable()
    if success:
        print("\\nBuild Complete!")
        print("Executable location: portable/dist/WhisperPortable.exe")
        print("\\nUsage Instructions:")
        print("1. Copy WhisperPortable.exe to any computer")
        print("2. Run WhisperPortable.exe (no installation required)")
        print("3. Enter your API key when prompted")
        print("4. Use the hotkeys to control recording")
        print("\\nThe executable is completely standalone and requires no dependencies!")
    else:
        print("\\nBuild Failed!")
        sys.exit(1)
