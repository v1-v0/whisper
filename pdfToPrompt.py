import os
import glob
import mlx_whisper
from pypdf import PdfReader
from pathlib import Path

# Ensure you have the 'source' folder created in the same directory as this script
# and place your PDF and Audio file inside it.

def get_prompt_from_source_folder(folder_path="./source"):
    """
    Scans the specified folder for PDFs, reads the first one found,
    and generates a prompt context.
    """
    # 1. Construct the search path (e.g., ./source/*.pdf)
    search_pattern = os.path.join(folder_path, "*.pdf")
    
    # 2. Find all PDF files
    pdf_files = glob.glob(search_pattern)
    
    # 3. Handle case where no PDF is found
    if not pdf_files:
        print(f"Warning: No PDF files found in folder '{folder_path}'. Using generic prompt.")
        return "Medical lecture context."
    
    # 4. Select the first PDF (Sorted alphabetically to ensure consistency)
    pdf_files.sort()
    target_pdf = pdf_files[0]
    print(f"--> Loading context from: {target_pdf}")
    
    # 5. Extract text content
    try:
        reader = PdfReader(target_pdf)
        text_content = ""
        
        # Read the first few pages (usually sufficient for vocabulary)
        # We limit to 5 pages to keep extraction fast
        for i, page in enumerate(reader.pages):
            if i > 4: break 
            text = page.extract_text()
            if text:
                text_content += text + " "
        
        # Clean up text (remove newlines and extra spaces)
        text_content = text_content.replace('\n', ' ').replace('  ', ' ')
        
        # Truncate to ~800 chars to fit Whisper's prompt token limit
        return f"Context: {text_content[:800]}"
        
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return "Medical lecture context."

def get_first_audio_file(folder_path="./source"):
    """
    Scans the folder for .mp3 or .m4a files and returns the first one found.
    """
    # Search for both extensions
    mp3_files = glob.glob(os.path.join(folder_path, "*.mp3"))
    mp3_files_upper = glob.glob(os.path.join(folder_path, "*.MP3")) # Case sensitive check
    m4a_files = glob.glob(os.path.join(folder_path, "*.m4a"))
    m4a_files_upper = glob.glob(os.path.join(folder_path, "*.M4A")) # Case sensitive check
    
    all_audio = mp3_files + mp3_files_upper + m4a_files + m4a_files_upper
    
    if not all_audio:
        return None
        
    all_audio.sort()
    return all_audio[0]

# --- Main Execution ---

# Define where your files are located
source_directory = "./source" 

# 1. Get the dynamic prompt from the first PDF
dynamic_prompt = get_prompt_from_source_folder(source_directory)
print(f"\nGenerated Prompt: {dynamic_prompt}\n")

# 2. Find the audio file automatically
audio_file_path = get_first_audio_file(source_directory)

if audio_file_path:
    print(f"--> Found audio file: {audio_file_path}")
    
    # 3. Run Transcription
    # In your pdfToPrompt.py script:
    
    result = mlx_whisper.transcribe(
        audio_file_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        initial_prompt=dynamic_prompt,
        verbose=True,
        # ADD THESE PARAMETERS:
        condition_on_previous_text=False, # Prevents the model from getting stuck on its own output
        compression_ratio_threshold=2.4,  # Filters out segments that are highly repetitive
        no_speech_threshold=0.6           # Skips segments that are likely silence
    )


    # Extract text
    raw_text = result["text"]

    # FIX: Ensure transcribed_text is strictly a string to satisfy Pylance
    if isinstance(raw_text, list):
        # If it happens to be a list, join it into a single string
        transcribed_text = " ".join([str(item) for item in raw_text])
    elif isinstance(raw_text, str):
        transcribed_text = raw_text
    else:
        # Fallback for unknown types
        transcribed_text = str(raw_text)

    print("-" * 30)
    print(transcribed_text)
    
    # 4. Save to Downloads folder
    # Get the user's home directory path
    home_dir = Path.home()
    downloads_path = home_dir / "Downloads"
    
    # Create a filename based on the audio filename
    audio_filename = os.path.basename(audio_file_path)
    text_filename = os.path.splitext(audio_filename)[0] + "_transcription.txt"
    output_file_path = downloads_path / text_filename
    
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(transcribed_text)
        print(f"\n[SUCCESS] Transcription saved to: {output_file_path}")
    except Exception as e:
        print(f"\n[ERROR] Could not save file: {e}")

else:
    print(f"Error: No .mp3 or .m4a files found in {source_directory}")