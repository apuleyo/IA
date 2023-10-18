import os
import shutil
import glob
import gc 
import whisperx
from whisperx.utils import WriteSRT


#List files 
myWhisperIn=r"C:\05 Python\WhisperX\Scripts\Files\mywhisperlistado.txt"
myWhisperOut=r"C:\05 Python\WhisperX\Scripts\Files\txt"
output_files = []
ext = [".3g2", ".3gp", ".asf", ".asx", ".avi", ".flv", \
                        ".m2ts", ".mkv", ".mov", ".mp4", ".mpg", ".mpeg", \
                        ".rm", ".swf", ".vob", ".wmv"]

def process_file(file_path, subdir):
 # print("\nSubdir: {subdir}\n")
    print(file_path)
    device = "cpu" 
    audio_file = file_path
    batch_size = 4 # reduce if low on GPU mem
    compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    # model = whisperx.load_model("medium", device, compute_type=compute_type, download_root="/data/local_models")
    # model= whisperx.load_model("/data/models/faster-whisper-medium.en", device, compute_type=compute_type)
    # model= whisperx.load_model("/usr/bin/fast-whisper-large-v2", device, compute_type=compute_type)
    model= whisperx.load_model(r"C:\05 Python\Models\models--guillaumekln--faster-whisper-large-v2\snapshots\f541c54c566e32dc1fbce16f98df699208837e8b", device, compute_type=compute_type)
    #model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=r"C:\05 Python\Models")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    #print(result["segments"]) # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    align_language = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    
     # Export to .txt file
    base_name = os.path.splitext(audio_file)[0]
    txt_output = f"{base_name}.txt"
    with open(txt_output, "w", encoding="utf-8") as txt:
        for segment in align_language["segments"]:
            txt.write(segment["text"] + "\n")

    # Export to .srt file
    srt_output = f"{base_name}.srt"
    with open(srt_output, "w", encoding="utf-8") as srt:
        writesrt = WriteSRT(".") #output file directory
        result["language"] = align_language
        writesrt.write_result(result, srt, {"max_line_width": None, "max_line_count": 2, "highlight_words": False, "preserve_segments": True})

    return txt_output, srt_output

   


    #print(result["segments"]) # after alignment



    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    #diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

    # add min/max number of speakers if known
    #diarize_segments = diarize_model(audio)
    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    #result = whisperx.assign_word_speakers(diarize_segments, result)
    #print(diarize_segments)
    #print(result["segments"]) # segments are now assigned speaker IDs    


def process_files(file_paths):
    for file_path in file_paths:
        txt_file, srt_file = process_file(file_path)
    #optional ostprocess_files(".")

def process_directory(directory):
    for subdir, dirs, files in os.walk(directory):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filename.endswith(tuple(ext)): #Only proccess files with allow extensions
                txt_file, srt_file = process_file(filepath, subdir)
                output_files.extend([txt_file, srt_file])
                print(f"Files {txt_file} and {srt_file} exports")
                subdirname=subdir[subdir.rfind("\\")+1:]             
                shutil.copy(txt_file, myWhisperOut+ os.sep + subdirname +"#" +  filename +".txt")
                 
                print(f"File {txt_file} copy to destination")
        #optional: postprocess_files(subdir)


def main():
    with open(myWhisperIn, encoding="utf8" ) as fileIn:
    # Reading from a file
        path=fileIn.readline()
        path=path.strip()
        while path != '':  # The EOF char is an empty string
            print(path, end='')
            if os.path.isdir(path): 
            # If path is a directory
                print(f"\nProcessing files in directory: {path}\n")
                process_directory(path)
            elif os.path.isfile(path):                                                  # If path is a file
                print(f"\nProcessing file: {path}\n")
                txt_file, srt_file = process_file(path, subdir=".")
            elif '*' in path:                                                           # If path is a wildcard
                files = glob.glob(path)
                print(f"\nProcessing files: {files}\n")
                process_files(files)
            else:                                                                       # If path is neither a file, nor a directory, nor a wildcard
                print("\nInvalid argument. Please enter a valid file path, wildcard, or directory.\n")        
            path=fileIn.readline()
            path=path.strip()

if __name__ == "__main__":
    main()