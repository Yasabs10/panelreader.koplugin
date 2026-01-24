#!/usr/bin/env python3
"""
Manga Processing Script
Processes manga folders/archives to generate panel data with Kumiko.

Usage:
    python process_manga.py [input_path]

Features:
- Creates Pages folder if needed
- Extracts CBZ/ZIP archives to separate folders
- Processes each folder with Kumiko
- Generates JSON files with normalized coordinates
- Supports RTL reading direction
"""

import os
import sys
import subprocess
import shutil
import zipfile
import argparse
from pathlib import Path

def create_kumiko_directories():
    """Create Pages and panel_result folders in Kumiko directory."""
    # Get the directory where this script is located (Kumiko folder)
    kumiko_dir = Path(__file__).parent
    pages_dir = kumiko_dir / "Pages"
    panel_result_dir = kumiko_dir / "panel_result"
    
    pages_dir.mkdir(exist_ok=True)
    panel_result_dir.mkdir(exist_ok=True)
    
    print(f"✅ Created/verified Kumiko directories:")
    print(f"   Pages: {pages_dir.absolute()}")
    print(f"   panel_result: {panel_result_dir.absolute()}")
    
    return pages_dir, panel_result_dir

def detect_file_type(file_path):
    """Detect actual file type using file command."""
    try:
        result = subprocess.run(['file', str(file_path)], 
                              capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        
        if 'Zip archive' in output:
            return 'zip'
        elif 'RAR archive' in output:
            return 'rar'
        elif '7-zip' in output:
            return '7z'
        elif 'tar' in output.lower():
            return 'tar'
        elif 'gzip compressed' in output:
            return 'gzip'
        else:
            print(f"🔍 File detection: {output}")
            return None
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  File type detection not available")
        return None

def extract_archive(archive_path, extract_to):
    """Extract archive with file type detection."""
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    
    # Detect actual file type
    detected_type = detect_file_type(archive_path)
    suffix = archive_path.suffix.lower()
    
    print(f"🔍 File: {archive_path.name} (ext: {suffix}, detected: {detected_type})")
    
    # Use detected type if available, otherwise fall back to extension
    if detected_type:
        archive_type = detected_type
    elif suffix in ['.cbz', '.zip']:
        archive_type = 'zip'
    elif suffix in ['.rar']:
        archive_type = 'rar'
    elif suffix in ['.gz', '.gzip']:
        archive_type = 'gzip'
    else:
        print(f"❌ Unsupported format: {suffix}")
        return False
    
    if archive_type == 'zip':
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"✅ Extracted {archive_path.name}")
            return True
        except Exception as e:
            print(f"❌ Failed to extract ZIP: {e}")
            if "not a zip file" in str(e):
                print(f"⚠️  File is not actually a ZIP archive despite .cbz extension!")
            return False
    
    elif archive_type == 'gzip':
        # Handle gzip compressed files (likely .tar.gz renamed to .cbz)
        try:
            # Try to extract as tar.gz first
            cmd = ['tar', '-xzf', str(archive_path), '-C', str(extract_to)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Extracted gzip/tar.gz {archive_path.name}")
            return True
        except subprocess.CalledProcessError:
            # If tar.gz fails, try just gzip decompression
            try:
                output_file = extract_to / archive_path.stem
                cmd = ['gunzip', '-c', str(archive_path)]
                with open(output_file, 'wb') as f:
                    subprocess.run(cmd, stdout=f, check=True)
                print(f"✅ Decompressed gzip {archive_path.name}")
                return True
            except Exception as e:
                print(f"❌ Failed to extract gzip: {e}")
                print(f"⚠️  File appears to be gzip but extraction failed")
                return False
    
    elif suffix in ['.rar']:
        # Extract RAR files using unrar
        try:
            cmd = ['unrar', 'x', str(archive_path), str(extract_to)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Extracted RAR {archive_path.name} to {extract_to}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to extract RAR {archive_path}: {e}")
            print(f"   Error output: {e.stderr}")
            return False
        except FileNotFoundError:
            print(f"❌ 'unrar' command not found. Please install unrar:")
            print(f"   Ubuntu/Debian: sudo apt install unrar")
            print(f"   Arch: sudo pacman -S unrar")
            return False
    
    elif suffix in ['.7z']:
        # Extract 7Z files using 7z
        try:
            cmd = ['7z', 'x', str(archive_path), f'-o{extract_to}', '-y']
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Extracted 7Z {archive_path.name} to {extract_to}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to extract 7Z {archive_path}: {e}")
            print(f"   Error output: {e.stderr}")
            return False
        except FileNotFoundError:
            print(f"❌ '7z' command not found. Please install p7zip:")
            print(f"   Ubuntu/Debian: sudo apt install p7zip-full")
            print(f"   Arch: sudo pacman -S p7zip")
            return False
    
    elif suffix in ['.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz']:
        # Extract TAR files using tar
        try:
            cmd = ['tar', '-xf', str(archive_path), '-C', str(extract_to)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Extracted TAR {archive_path.name} to {extract_to}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to extract TAR {archive_path}: {e}")
            print(f"   Error output: {e.stderr}")
            return False
    
    else:
        print(f"❌ Unsupported archive format: {suffix}")
        print(f"   Supported formats: .cbz, .zip, .rar, .7z, .tar, .tar.gz, .tgz, .tar.bz2, .tbz2, .tar.xz, .txz")
        return False

def is_archive(file_path):
    """Check if file is a supported archive."""
    return file_path.suffix.lower() in ['.cbz', '.zip', '.rar', '.7z', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz']

def process_image_with_kumiko(image_path, output_dir):
    """Process a single image with Kumiko to generate HTML output."""
    image_name = image_path.stem
    output_html = output_dir / f"{image_name}.html"
    
    # First try with HTML output
    success, html_file = try_kumiko_with_flags(image_path, output_html, ['--rtl', '--html'])
    
    if success:
        return True, html_file
    
    # Fallback: try without HTML flag (might generate JSON instead)
    print(f"   🔄 Trying fallback without HTML flag...")
    output_json = output_dir / f"{image_name}.json"
    success, json_file = try_kumiko_with_flags(image_path, output_json, ['--rtl'])
    
    if success and json_file.exists():
        # Convert JSON to HTML format for processing
        print(f"   🔄 Converting JSON to HTML format...")
        try:
            return convert_json_to_html(json_file, output_html)
        except Exception as e:
            print(f"   ❌ Failed to convert JSON to HTML: {e}")
            return False, None
    
    # Final fallback: try with minimal flags
    print(f"   🔄 Trying minimal flags...")
    success, file = try_kumiko_with_flags(image_path, output_html, [])
    
    return success, file if success else None

def try_kumiko_with_flags(image_path, output_file, flags):
    """Try running Kumiko with specific flags."""
    image_name = image_path.stem
    
    # Build Kumiko command with -i for input and -o for output
    cmd = ['python3', 'kumiko', '-i', str(image_path)] + flags + ['-o', str(output_file)]
    
    print(f"   Trying: {' '.join(cmd)}")
    
    try:
        # Change to Kumiko directory for execution
        original_cwd = os.getcwd()
        kumiko_dir = Path(__file__).parent
        os.chdir(kumiko_dir)
        
        # Run Kumiko
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            if output_file.exists():
                print(f"   ✅ Success with flags: {' '.join(flags)}")
                return True, output_file
            else:
                print(f"   ⚠️  File not created: {output_file}")
                print(f"   Kumiko output: {result.stdout}")
                return False, None
        else:
            print(f"   ❌ Failed with flags {' '.join(flags)}:")
            print(f"      Return code: {result.returncode}")
            if result.stderr:
                error_msg = result.stderr.strip()
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                print(f"      Error: {error_msg}")
            return False, None
            
    except subprocess.TimeoutExpired:
        print(f"   ❌ Timeout with flags {' '.join(flags)}")
        return False, None
    except Exception as e:
        print(f"   ❌ Exception with flags {' '.join(flags)}: {e}")
        return False, None
    finally:
        os.chdir(original_cwd)

def convert_json_to_html(json_file, html_file):
    """Convert Kumiko JSON output to HTML format for processing."""
    import json
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create a simple HTML structure with the panel data
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Kumiko Panel Data</title>
</head>
<body>
    <h1>Panel Data for {json_file.stem}</h1>
    <script>
        var panelData = {json.dumps(data, indent=2)};
    </script>
</body>
</html>"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"   ✅ Converted JSON to HTML: {html_file}")
        return True, html_file
        
    except Exception as e:
        print(f"   ❌ JSON to HTML conversion failed: {e}")
        return False, None

def combine_htmls_to_json(html_files, output_json):
    """Combine multiple HTML files into a single JSON with page-based structure."""
    import json
    import re
    
    pages_data = []
    reading_direction = "rtl"  # Default to RTL for manga
    
    print(f"🔄 Combining {len(html_files)} HTML files to JSON...")
    
    # First, check which HTML files actually exist
    existing_html_files = []
    for html_file in html_files:
        if html_file.exists():
            existing_html_files.append(html_file)
            print(f"   Found: {html_file.name}")
        else:
            print(f"   ❌ Missing: {html_file}")
    
    if not existing_html_files:
        print(f"❌ No HTML files found to process")
        return False
    
    # Debug: Show content of first HTML file
    if existing_html_files:
        first_html = existing_html_files[0]
        print(f"🔍 Debug: First HTML file content preview:")
        try:
            with open(first_html, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"   Size: {len(content)} characters")
                print(f"   First 500 chars: {content[:500]}")
                print(f"   Contains 'panel': {'panel' in content.lower()}")
                print(f"   Contains 'json': {'json' in content.lower()}")
                print(f"   Contains 'coordinates': {'coordinates' in content.lower()}")
        except Exception as e:
            print(f"   Error reading file: {e}")
    
    # Sort HTML files by name to ensure correct page order
    existing_html_files.sort(key=lambda x: x.name)
    
    for page_num, html_file in enumerate(existing_html_files, 1):
        print(f"   Processing page {page_num}: {html_file.name}")
        
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Debug: Show what we're looking for
            print(f"     File size: {len(html_content)} chars")
            
            # Try to extract panel data from HTML using multiple patterns
            panel_patterns = [
                # JSON format in script tags - look for the actual panels array (full array)
                r'"panels":\s*(\[[^\]]*(?:\][^\]]*)*\])',
                r'"panels":\s*(\[\s*\[[^\]]*\]\s*(?:,\s*\[[^\]]*\]\s*)*\])',
                # Script tag JSON
                r'<script[^>]*>.*?var\s+\w+\s*=\s*(\{.*?\});.*?</script>',
                r'<script[^>]*>.*?const\s+\w+\s*=\s*(\{.*?\});.*?</script>',
                r'<script[^>]*>.*?let\s+\w+\s*=\s*(\{.*?\});.*?</script>',
                # Direct JSON in HTML
                r'(\{[^}]*"panels"[^}]*\})',
                r'(\{[^}]*"x"[^}]*"y"[^}]*"w"[^}]*"h"[^}]*\})',
                # Panel coordinates in various formats
                r'panel.*?{.*?x.*?(\d+\.?\d*).*?y.*?(\d+\.?\d*).*?w.*?(\d+\.?\d*).*?h.*?(\d+\.?\d*)}',
                r'x.*?(\d+\.?\d*).*?y.*?(\d+\.?\d*).*?width.*?(\d+\.?\d*).*?height.*?(\d+\.?\d*)',
                r'"x":\s*(\d+\.?\d*),\s*"y":\s*(\d+\.?\d*),\s*"w":\s*(\d+\.?\d*),\s*"h":\s*(\d+\.?\d*)',
                r'x:\s*(\d+\.?\d*),\s*y:\s*(\d+\.?\d*),\s*w:\s*(\d+\.?\d*),\s*h:\s*(\d+\.?\d*)',
                # Array format - capture nested arrays (multiple)
                r'\[\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*(?:,\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]\s*)*',
                r'\[\s*\{\s*"x"\s*:\s*(\d+\.?\d*)\s*,\s*"y"\s*:\s*(\d+\.?\d*)\s*,\s*"w"\s*:\s*(\d+\.?\d*)\s*,\s*"h"\s*:\s*(\d+\.?\d*)\s*\}\s*\]'
            ]
            
            matches = []
            used_pattern = None
            
            for i, pattern in enumerate(panel_patterns):
                pattern_matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
                if pattern_matches:
                    matches = pattern_matches
                    used_pattern = f"Pattern {i+1}: {pattern[:50]}..."
                    print(f"     Found {len(matches)} matches with {used_pattern}")
                    
                    # Debug: Show first match
                    if pattern_matches:
                        first_match = str(pattern_matches[0])
                        if len(first_match) > 200:
                            first_match = first_match[:200] + "..."
                        print(f"     First match: {first_match}")
                    break
            
            if not matches:
                print(f"     ❌ No panel matches found")
                # Try to find any JSON data
                json_patterns = [
                    r'(\{[^{}]*\})',
                    r'(\[[^\[\]]*\])'
                ]
                for pattern in json_patterns:
                    json_matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
                    if json_matches:
                        print(f"     Found {len(json_matches)} JSON-like structures")
                        for i, match in enumerate(json_matches[:3]):  # Show first 3
                            match_str = str(match)
                            if len(match_str) > 100:
                                match_str = match_str[:100] + "..."
                            print(f"       JSON {i+1}: {match_str}")
                # Create empty page data and continue
                page_data = {
                    "page": page_num,
                    "image": html_file.stem + ".jpg",  # Default image name
                    "panels": []
                }
                pages_data.append(page_data)
                continue
            
            # Extract image dimensions from HTML - prioritize actual image size
            img_patterns = [
                # Look for "size": [width, height] in the JSON
                r'"size":\s*\[(\d+)\s*,\s*(\d+)\]',
                r'"size":\s*\[(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\]',
                # Alternative formats
                r'"width":\s*(\d+),\s*"height":\s*(\d+)',
                r'width:\s*(\d+),\s*height:\s*(\d+)',
                r'(\d+)\s*x\s*(\d+)',  # e.g., "800x600"
                # CSS dimensions (lower priority)
                r'width.*?(\d+).*?height.*?(\d+)'
            ]
            
            img_w, img_h = 800, 1200  # Default dimensions
            for pattern in img_patterns:
                img_matches = re.findall(pattern, html_content, re.IGNORECASE)
                if img_matches:
                    img_w, img_h = map(float, img_matches[0])
                    print(f"     Image dimensions: {img_w}x{img_h} (from pattern: {pattern[:30]}...)")
                    break
            
            # Convert matches to panel format
            page_panels = []
            panels_added = 0
            
            for match in matches:
                try:
                    # Handle different match formats
                    if isinstance(match, str):
                        # Check if it's a panels array like [[430, 56, 160, 291], [231, 56, 192, 2...]
                        if match.strip().startswith('[') and not match.strip().startswith('{'):
                            # Parse as panels array
                            try:
                                panels_array = json.loads(match)
                                if isinstance(panels_array, list):
                                    for panel_coords in panels_array:
                                        if isinstance(panel_coords, list) and len(panel_coords) >= 4:
                                            x, y, w, h = map(float, panel_coords[:4])
                                            panels_added += add_normalized_panel_to_page(page_panels, x, y, w, h, img_w, img_h)
                                print(f"     Processed {len(panels_array)} panels from array")
                            except json.JSONDecodeError:
                                # Try regex extraction for nested arrays
                                array_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
                                coord_matches = re.findall(array_pattern, match)
                                for x, y, w, h in coord_matches:
                                    panels_added += add_normalized_panel_to_page(page_panels, float(x), float(y), float(w), float(h), img_w, img_h)
                                print(f"     Processed {len(coord_matches)} panels via regex")
                        else:
                            # Try to parse as JSON first
                            try:
                                data = json.loads(match)
                                if 'panels' in data:
                                    panel_list = data['panels']
                                    if isinstance(panel_list, list):
                                        for panel_data in panel_list:
                                            if isinstance(panel_data, list) and len(panel_data) >= 4:
                                                x, y, w, h = map(float, panel_data[:4])
                                                panels_added += add_normalized_panel_to_page(page_panels, x, y, w, h, img_w, img_h)
                                            elif isinstance(panel_data, dict) and all(key in panel_data for key in ['x', 'y', 'w', 'h']):
                                                x, y, w, h = map(float, [panel_data['x'], panel_data['y'], panel_data['w'], panel_data['h']])
                                                panels_added += add_normalized_panel_to_page(page_panels, x, y, w, h, img_w, img_h)
                                    print(f"     Processed {len(panel_list)} panels from JSON")
                                elif 'x' in data and 'y' in data:
                                    panel_list = [data]
                                    for panel_data in panel_list:
                                        if all(key in panel_data for key in ['x', 'y', 'w', 'h']):
                                            x, y, w, h = map(float, [panel_data['x'], panel_data['y'], panel_data['w'], panel_data['h']])
                                            panels_added += add_normalized_panel_to_page(page_panels, x, y, w, h, img_w, img_h)
                            except json.JSONDecodeError:
                                # Try regex extraction
                                coord_matches = re.findall(r'(\d+\.?\d*)', match)
                                if len(coord_matches) >= 4:
                                    # Process in groups of 4
                                    for i in range(0, len(coord_matches), 4):
                                        if i + 3 < len(coord_matches):
                                            x, y, w, h = map(float, coord_matches[i:i+4])
                                            panels_added += add_normalized_panel_to_page(page_panels, x, y, w, h, img_w, img_h)
                    else:
                        # Tuple format from regex
                        x, y, w, h = map(float, match[:4])
                        panels_added += add_normalized_panel_to_page(page_panels, x, y, w, h, img_w, img_h)
                        
                except Exception as e:
                    print(f"     Error processing match: {e}")
                    continue
            
            # Create page data structure
            page_data = {
                "page": page_num,
                "image": html_file.stem + ".jpg",  # Use HTML file name as image name
                "panels": page_panels
            }
            pages_data.append(page_data)
            
            print(f"     Added {panels_added} panels for page {page_num}")
        
        except Exception as e:
            print(f"   ❌ Error processing {html_file}: {e}")
            # Create empty page data even on error
            page_data = {
                "page": page_num,
                "image": html_file.stem + ".jpg",
                "panels": []
            }
            pages_data.append(page_data)
            continue
    
    if not pages_data:
        print(f"❌ No page data created")
        return False
    
    # Count total panels
    total_panels = sum(len(page_data["panels"]) for page_data in pages_data)
    print(f"📊 Total pages: {len(pages_data)}, Total panels: {total_panels}")
    
    # Create final JSON structure with pages array
    json_data = {
        "reading_direction": reading_direction,
        "total_pages": len(pages_data),
        "pages": pages_data
    }
    
    # Write JSON output
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Combined {total_panels} panels from {len(pages_data)} pages to {output_json}")
        return True
        
    except Exception as e:
        print(f"❌ Error writing JSON: {e}")
        return False

def add_normalized_panel_to_page(page_panels, x, y, w, h, img_w, img_h):
    """Add a normalized panel to a specific page's panel list."""
    # Normalize to 0-1 range
    normalized_x = x / img_w
    normalized_y = y / img_h
    normalized_w = w / img_w
    normalized_h = h / img_h
    
    # Clamp to valid range
    normalized_x = max(0, min(1, normalized_x))
    normalized_y = max(0, min(1, normalized_y))
    normalized_w = max(0, min(1, normalized_w))
    normalized_h = max(0, min(1, normalized_h))
    
    panel = {
        "x": round(normalized_x, 3),
        "y": round(normalized_y, 3),
        "w": round(normalized_w, 3),
        "h": round(normalized_h, 3)
    }
    
    page_panels.append(panel)
    print(f"       Added panel: x={panel['x']}, y={panel['y']}, w={panel['w']}, h={panel['h']}")
    return 1

def add_normalized_panel(all_panels, x, y, w, h, img_w, img_h):
    """Add a normalized panel to the list."""
    # Normalize to 0-1 range
    normalized_x = x / img_w
    normalized_y = y / img_h
    normalized_w = w / img_w
    normalized_h = h / img_h
    
    # Clamp to valid range
    normalized_x = max(0, min(1, normalized_x))
    normalized_y = max(0, min(1, normalized_y))
    normalized_w = max(0, min(1, normalized_w))
    normalized_h = max(0, min(1, normalized_h))
    
    panel = {
        "x": round(normalized_x, 3),
        "y": round(normalized_y, 3),
        "w": round(normalized_w, 3),
        "h": round(normalized_h, 3)
    }
    
    all_panels.append(panel)
    print(f"       Added panel: x={panel['x']}, y={panel['y']}, w={panel['w']}, h={panel['h']}")
    return 1

def process_with_kumiko(folder_path, output_dir):
    """Process a folder with Kumiko by processing each image separately."""
    folder_name = folder_path.name
    output_json = output_dir / f"{folder_name}.json"
    temp_html_dir = output_dir / f"{folder_name}_temp"
    temp_html_dir.mkdir(exist_ok=True)
    
    print(f"🔄 Processing folder {folder_name} with individual image processing...")
    
    # Get all image files in the folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(folder_path.glob(f"*{ext}"))
        image_files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    # Sort files for consistent processing
    image_files.sort()
    
    if not image_files:
        print(f"❌ No image files found in {folder_path}")
        return False
    
    print(f"   Found {len(image_files)} image files")
    
    # Process each image separately
    html_files = []
    successful_images = 0
    
    for image_file in image_files:
        success, html_file = process_image_with_kumiko(image_file, temp_html_dir)
        if success and html_file and html_file.exists():
            html_files.append(html_file)
            successful_images += 1
        else:
            print(f"   ⚠️  Failed to process {image_file.name}")
    
    print(f"   Successfully processed {successful_images}/{len(image_files)} images")
    
    if not html_files:
        print(f"❌ No HTML files were generated")
        return False
    
    # List all HTML files that were actually created
    print(f"   HTML files created:")
    for html_file in html_files:
        print(f"     - {html_file.name}")
    
    # Combine all HTML files into single JSON
    success = combine_htmls_to_json(html_files, output_json)
    
    # Clean up temporary HTML files
    try:
        import shutil
        shutil.rmtree(temp_html_dir)
        print(f"🧹 Cleaned up temporary files")
    except Exception as e:
        print(f"⚠️  Could not clean up temp files: {e}")
    
    return success

def process_input(input_path, pages_dir, panel_result_dir):
    """Process input path (folder or archive)."""
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"❌ Input path does not exist: {input_path}")
        return False
    
    if is_archive(input_path):
        # Extract archive first
        extract_folder = pages_dir / input_path.stem
        extract_folder.mkdir(exist_ok=True)
        
        print(f"📦 Processing archive: {input_path}")
        if not extract_archive(input_path, extract_folder):
            return False
        
        # Process extracted folder
        return process_with_kumiko(extract_folder, panel_result_dir)
        
    elif input_path.is_dir():
        # Process folder directly
        print(f"📁 Processing folder: {input_path}")
        return process_with_kumiko(input_path, panel_result_dir)
        
    else:
        print(f"❌ Unsupported input type: {input_path}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process manga folders/archives with Kumiko")
    parser.add_argument('input', help='Input folder or archive file')
    parser.add_argument('--pages-dir', default='Pages', help='Pages directory name (within Kumiko)')
    parser.add_argument('--output-dir', default='panel_result', help='Output directory name (within Kumiko)')
    
    args = parser.parse_args()
    
    print("🚀 Manga Processing Script Started")
    print("=" * 50)
    
    # Create directories in Kumiko folder
    pages_dir, panel_result_dir = create_kumiko_directories()
    
    # Process input
    success = process_input(args.input, pages_dir, panel_result_dir)
    
    print("=" * 50)
    if success:
        print("🎉 Processing completed successfully!")
        print(f"📂 Results in: {panel_result_dir.absolute()}")
    else:
        print("❌ Processing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
