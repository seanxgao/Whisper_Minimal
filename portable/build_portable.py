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

def build_portable(preset: str = "simple", output_name: str = None):
    """
    Build portable executable directly from Minimal codebase with specified preset
    
    Args:
        preset: Preset configuration name (simple, professional, fast)
        output_name: Custom output executable name (optional)
    """
    
    print("=== Portable Builder ===")
    print(f"Building with preset: {preset}")
    
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
            "whisper_minimal.py",
            "config_utils.py", 
            "recorder.py",
            "transcriber.py",
            "text_cleaner.py",
            "keyboard_typer.py",
            "requirements.txt"
        ]
        
        for file_name in files_to_copy:
            src_file = minimal_dir / file_name
            if src_file.exists():
                shutil.copy2(src_file, build_dir / file_name)
                print(f"  - Copied {file_name}")
            else:
                print(f"  - Warning: {file_name} not found")
        
        # Copy vad directory
        vad_src = minimal_dir / "vad"
        vad_dst = build_dir / "vad"
        if vad_src.exists():
            shutil.copytree(vad_src, vad_dst)
            print(f"  - Copied vad/ directory")
        
        # Copy realtime directory
        realtime_src = minimal_dir / "realtime"
        realtime_dst = build_dir / "realtime"
        if realtime_src.exists():
            shutil.copytree(realtime_src, realtime_dst)
            print(f"  - Copied realtime/ directory")
        
        # Create portable main script with preset
        create_portable_main(build_dir, preset)
        
        # Determine output name
        if not output_name:
            output_name = f"WhisperPortable_{preset.capitalize()}"
        
        # Build executable from temp directory
        print("Building standalone executable...")
        success = build_executable(build_dir, portable_dir, output_name)
        
        if success:
            print("Build successful!")
            print(f"Executable location: {portable_dir / 'dist' / f'{output_name}.exe'}")
            
            # Clean up temp build directory
            cleanup_temp_files(portable_dir)
            
            return True
        else:
            print("Build failed!")
            return False
            
    except Exception as e:
        print(f"Error during build: {e}")
        return False

def create_portable_main(build_dir, preset: str):
    """Create the main portable script with preset configuration"""
    
    main_script = build_dir / "whisper_portable.py"
    
    script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whisper Portable - Standalone executable version
Built with preset: {preset}
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
    """Main entry point - portable version with {preset} preset"""
    print("=== Whisper Portable ({preset.capitalize()}) ===")
    print("Standalone portable version")
    print("Press Ctrl+C to exit")
    print()

    try:
        # Get API key
        api_key = get_api_key()

        # Set environment variable for the app
        os.environ["OPENAI_API_KEY"] = api_key

        # Import and run the main app with preset
        from whisper_minimal import App
        app = App(preset="{preset}")
        app.start()
    except KeyboardInterrupt:
        print("\\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Application error: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    with open(main_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"  - Created portable main script with {preset} preset")

def build_executable(build_dir, portable_dir, output_name: str):
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
            "--name", output_name,          # Executable name
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Whisper Portable with preset configuration")
    parser.add_argument("--preset", "-p", 
                       choices=["simple", "professional", "fast"],
                       default="simple",
                       help="Preset configuration to use (default: simple)")
    parser.add_argument("--output", "-o", 
                       help="Custom output executable name (optional)")
    parser.add_argument("--list-presets", "-l", 
                       action="store_true",
                       help="List available presets and exit")
    
    args = parser.parse_args()
    
    # List presets if requested
    if args.list_presets:
        print("Available Presets:")
        presets = {
            "simple": "Basic configuration for daily use",
            "professional": "Advanced configuration for professional users", 
            "fast": "Speed-optimized configuration with reduced processing"
        }
        for name, description in presets.items():
            print(f"  {name}: {description}")
        sys.exit(0)
    
    success = build_portable(preset=args.preset, output_name=args.output)
    if success:
        output_name = args.output or f"WhisperPortable_{args.preset.capitalize()}"
        print("\\nBuild Complete!")
        print(f"Executable location: portable/dist/{output_name}.exe")
        print("\\nUsage Instructions:")
        print(f"1. Copy {output_name}.exe to any computer")
        print(f"2. Run {output_name}.exe (no installation required)")
        print("3. Enter your API key when prompted")
        print("4. Use the hotkeys to control recording")
        print(f"\\nThe executable uses the '{args.preset}' preset configuration!")
    else:
        print("\\nBuild Failed!")
        sys.exit(1)
