import os
import glob
import mlx_whisper
from pypdf import PdfReader

# Ensure you have the 'source' folder created in the same directory as this script
# and place your PDF file inside it.

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

# --- Main Execution ---

# Define where your files are located
source_directory = "./source" 
audio_file_path = "your_audio_file.mp3" # Update this to your specific audio file name

# Get the dynamic prompt from the first PDF in the source folder
dynamic_prompt = get_prompt_from_source_folder(source_directory)

print(f"\nGenerated Prompt: {dynamic_prompt}\n")

# Run Transcription
# Note: Ensure you have an audio file at 'audio_file_path' before running
if os.path.exists(audio_file_path):
    result = mlx_whisper.transcribe(
        audio_file_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
        initial_prompt=dynamic_prompt,
        verbose=True
    )

    print("-" * 30)
    print(result["text"])
else:
    print(f"Error: Audio file not found at {audio_file_path}")