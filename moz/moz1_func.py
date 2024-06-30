# @title 起動、出入力先の指定、Transcribe！
import os
import json
from pydub import AudioSegment
from faster_whisper import WhisperModel
import torch
from tqdm import tqdm
import ipywidgets as widgets
from IPython.display import display

def get_audio_duration(file_path):
    try:
        if not os.path.exists(file_path):
            return "File does not exist"
        audio = AudioSegment.from_file(file_path)
        duration = len(audio) / 1000.0
        return duration
    except Exception as e:
        return f"An error occurred: {str(e)}"

def format_timestamp(seconds):
    hrs, secs = divmod(seconds, 3600)
    mins, secs = divmod(secs, 60)
    millis = int((secs % 1) * 1000)
    return f"{int(hrs):02}:{int(mins):02}:{int(secs):02},{millis:03}"

def transcribe_audio(input_folder, output_folder):
    model = WhisperModel("large-v2", device="cuda", compute_type="float32")

    main_files = []

    for file_name in os.listdir(input_folder):
        if file_name.endswith((".mp3", ".mp4", ".mkv", ".webm")):
            file_path = os.path.join(input_folder, file_name)
            try:
                print("いろいろ読み込み中です･･･")
                segments, _ = model.transcribe(file_path, word_timestamps=True, beam_size=5, initial_prompt="Hello.I am Scott.", language="en", vad_filter=True)
                print(f"{file_name}の文字起こしを開始します")
                total_duration = get_audio_duration(file_path)
                # Initialize tqdm progress bar
                progress_bar = tqdm(total=total_duration, unit="s", position=0, leave=True)
                words_data = []
                last_update_time = 0

                for segment in segments:

                    # Collect words data
                    for word in segment.words:
                        word_info = {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word
                        }
                        words_data.append(word_info)
                        progress_bar.update(segment.end - last_update_time)
                        last_update_time = segment.end

                progress_bar.update(total_duration - last_update_time)
                # Close tqdm progress bar
                progress_bar.close()
                print("文字起こし1つ終了")
                # Clean up word strings
                for word_info in words_data:
                    word_info["word"] = word_info["word"].replace(" Dr.", " Dr★").replace(" dr.", " dr★")

                # Remove special character markers
                cleaned_words_data = []
                for word_info in words_data:
                    cleaned_word_info = {
                        "start": word_info["start"],
                        "end": word_info["end"],
                        "word": word_info["word"].replace("★", "")
                    }
                    cleaned_words_data.append(cleaned_word_info)

                input_file_name = os.path.splitext(file_name)[0]

                # JSON output
                json_output_file_name = f"{input_file_name}.json"
                json_output_path = os.path.join(output_folder, json_output_file_name)
                with open(json_output_path, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_words_data, f, ensure_ascii=False, indent=4)
                main_files.append(json_output_path)

                # SRT output
                srt_entries = []
                entry_number = 1
                segment_text = ""
                segment_start = None
                segment_end = None

                for word_info in words_data:
                    if segment_start is None:
                        segment_start = word_info["start"]
                    segment_text += word_info["word"]
                    segment_end = word_info["end"]

                    if word_info["word"].endswith('.'):
                        srt_entries.append({
                            "number": entry_number,
                            "start": segment_start,
                            "end": segment_end,
                            "text": segment_text.strip()
                        })
                        entry_number += 1
                        segment_text = ""
                        segment_start = None

                if segment_text.strip():
                    srt_entries.append({
                        "number": entry_number,
                        "start": segment_start,
                        "end": segment_end,
                        "text": segment_text.strip()
                    })

                srt_output_file_name = f"{input_file_name}.srt"
                srt_output_path = os.path.join(output_folder, srt_output_file_name)
                with open(srt_output_path, 'w', encoding='utf-8') as f:
                    for entry in srt_entries:
                        start_time = format_timestamp(entry["start"])
                        end_time = format_timestamp(entry["end"])
                        text = entry['text'].replace(" Dr★", " Dr.").replace(" dr★", " dr.").replace("Dr★", "Dr.")
                        f.write(f"{entry['number']}\n{start_time} --> {end_time}\n{text}\n\n")
                main_files.append(srt_output_path)

                # Text files output
                txt_nr_content = ""
                for word_info in words_data:
                    if not txt_nr_content:
                        txt_nr_content += word_info['word'].lstrip()
                    else:
                        txt_nr_content += word_info['word']

                txt_nr_output_file_name = f"{input_file_name}_NR.txt"
                txt_nr_output_path = os.path.join(output_folder, txt_nr_output_file_name)
                with open(txt_nr_output_path, 'w', encoding='utf-8') as f:
                    txt_nr_content = txt_nr_content.replace(" Dr★", " Dr.").replace(" dr★", " dr.").replace("Dr★", "Dr.")
                    f.write(txt_nr_content)
                main_files.append(txt_nr_output_path)

                txt_r_content = ""
                previous_word_end = 0
                is_first_word = True
                for word in words_data:
                    if is_first_word or txt_r_content.endswith("\n"):
                        txt_r_content += word['word'].strip()
                    else:
                        txt_r_content += word['word']

                    if "." in word['word']:
                        if word['start'] - previous_word_end >= 0.5:
                            txt_r_content += "\n"
                        previous_word_end = word['end']
                    is_first_word = False

                txt_r_output_file_name = f"{input_file_name}_R.txt"
                txt_r_output_path = os.path.join(output_folder, txt_r_output_file_name)
                with open(txt_r_output_path, 'w', encoding='utf-8') as f:
                    txt_r_content = txt_r_content.replace(" Dr★", " Dr.").replace(" dr★", " dr.").replace("Dr★", "Dr.")
                    f.write(txt_r_content)
                main_files.append(txt_r_output_path)

            except Exception as e:
                print(f"Error processing file {file_name}: {str(e)}")

    print("全部終わったよ！")
    return main_files
    # Memory cleanup
torch.cuda.empty_cache()

# ウィジェットの作成
input_folder_widget = widgets.Text(
    value='/content/drive/MyDrive/My Input',
    placeholder='Enter input folder path',
    description='Input Folder:',
    disabled=False
)

output_folder_widget = widgets.Text(
    value='/content/drive/MyDrive/My Output',
    placeholder='Enter output folder path',
    description='Output Folder:',
    disabled=False
)

transcribe_button = widgets.Button(description="Transcribe")

# ボタンがクリックされたときの処理
def on_button_click(b):
    input_folder = input_folder_widget.value
    output_folder = output_folder_widget.value
    result_files = transcribe_audio(input_folder, output_folder)
    print("Output files:")
    for file_path in result_files:
        print(file_path)

def clk():
    transcribe_button.on_click(on_button_click)

def dp():
    display(input_folder_widget)
    display(output_folder_widget)
    display(transcribe_button)
