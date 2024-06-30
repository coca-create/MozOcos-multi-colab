from moz import moz1_func as m1
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

transcribe_button.on_click(on_button_click)

display(input_folder_widget)
display(output_folder_widget)
display(transcribe_button)
