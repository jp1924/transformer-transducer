import multiprocessing as mp
from multiprocessing import Queue

import gradio as gr
import librosa
import numpy as np
import tritonclient.grpc as grpcclient


SAMPLE_RATE = 16000


with gr.Blocks() as demo:
    triton_client = grpcclient.InferenceServerClient(
        url="0.0.0.0:9843",
        verbose=False,
        ssl=False,
        root_certificates=None,
        private_key=None,
        certificate_chain=None,
    )

    def record(audio_filepath: str, buffer: np.ndarray):
        audio, sr = librosa.load(audio_filepath, sr=SAMPLE_RATE)
        buffer = np.concatenate((buffer, audio))

        while len(buffer) >= sr:
            one_second_audio = buffer[:sr]
            audio_queue.put(one_second_audio)
            buffer = buffer[sr:]

        return buffer

    def send_to_server():
        cache, text = np.zeros((12, 512, 1, 512), dtype=np.float32), np.array([], dtype=np.int32)
        while True:
            if not audio_queue.qsize():
                continue
            audio: np.ndarray = audio_queue.get()
            if isinstance(audio, str) and audio == "end_signal":
                cache, text = np.zeros((12, 512, 1, 512), dtype=np.float32), np.array([], dtype=np.int32)
                continue

            audio = audio.astype(np.float32)

            recv_result = triton_client.infer(
                model_name="transformer_transducer",
                inputs=[
                    grpcclient.InferInput("audio", audio.shape, "FP32").set_data_from_numpy(audio),
                    grpcclient.InferInput("cache", cache.shape, "FP32").set_data_from_numpy(cache),
                    grpcclient.InferInput("text", text.shape, "INT32").set_data_from_numpy(text),
                ],
                outputs=[grpcclient.InferRequestedOutput("text"), grpcclient.InferRequestedOutput("cache")],
            )
            recv_text = recv_result.as_numpy("text")
            cache = recv_result.as_numpy("cache")

            if not recv_text.all():
                text = recv_text
            else:
                text = np.concatenate((text, recv_text))

            text_queue.put(recv_text.tolist())
            # text_queue.put([0])

    def recv_from_server(buffer: list):
        if text_queue.qsize():
            text_ls = text_queue.get()
            buffer = text_ls

        # return buffer, tokenizer.decode(buffer, skip_special_tokens=True)
        return buffer, str(buffer)

    audio_queue = Queue()
    audio, audio_buffer = gr.Audio(type="filepath", streaming=True), gr.State(value=np.array([]))

    text_queue = Queue()
    text, text_buffer = gr.Textbox(value=""), gr.State(value=[])

    mp.Process(target=send_to_server).start()
    audio.stream(fn=record, inputs=[audio, audio_buffer], outputs=[audio_buffer])
    audio.stop_recording(fn=lambda: audio_queue.put("end_signal"))
    demo.load(recv_from_server, inputs=[text_buffer], outputs=[text_buffer, text], every=0.1)

if __name__ == "__main__":
    demo.launch(debug=True)
