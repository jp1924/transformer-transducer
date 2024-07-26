import multiprocessing as mp
from multiprocessing import Queue

import gradio as gr
import librosa
import numpy as np


SAMPLE_RATE = 16000

with gr.Blocks() as demo:

    def record(audio_filepath: str, buffer: np.ndarray):
        audio, sr = librosa.load(audio_filepath, sr=SAMPLE_RATE)
        buffer = np.concatenate((buffer, audio))

        # audio_filepath = Path(audio_filepath)
        # shutil.rmtree(audio_filepath.parent)

        while len(buffer) >= sr:
            # print(f"put: {len(buffer) / sr}")
            one_second_audio = buffer[:sr]
            audio_queue.put(one_second_audio.tolist())
            buffer = buffer[sr:]
        # print(len(buffer) / sr)
        return buffer

    def send_to_server():
        while True:
            if not audio_queue.qsize():
                continue
            audio = audio_queue.get()

            # push to api

            text_queue.put([0])

    def recv_from_server(buffer: list):
        if text_queue.qsize():
            text_ls = [text_queue.get() for _ in range(text_queue.qsize())]
            buffer.extend(sum(text_ls, []))
            # tokenizer.decode(buffer, skip_special_tokens=True)

        return buffer, str(buffer)

    audio_queue = Queue()
    audio_buffer = gr.State(value=np.array([]))
    audio = gr.Audio(type="filepath", streaming=True)

    text_queue = Queue()
    text_buffer = gr.State(value=[])
    text = gr.Textbox(value="")

    mp.Process(target=send_to_server).start()
    audio.stream(fn=record, inputs=[audio, audio_buffer], outputs=[audio_buffer])
    demo.load(recv_from_server, inputs=[text_buffer], outputs=[text_buffer, text], every=0.1)

if __name__ == "__main__":
    demo.launch(debug=True)
