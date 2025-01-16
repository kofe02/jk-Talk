import requests
import json
import io
import time
import threading
import readchar
import pyaudio
from voicevox_core import VoicevoxCore, METAS
from pathlib import Path
import wave
import simpleaudio as sa
from faster_whisper import WhisperModel

model = WhisperModel("kotoba-tech/kotoba-whisper-v2.0-faster")
CHUNK = 2**10
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
record_time = 5
output_path = "./output.wav"

PROTCOL = "http"
HOST = "localhost"
PORT = "11434"
HEADERS = {"content-type": "application/json"}
URL = f"{PROTCOL}://{HOST}:{PORT}/api/chat"
MODEL = "dsasai/llama3-elyza-jp-8b:latest"
SPREAKER_ID = 5 #ずんだもん
open_jtalk_dict_dir = Path("open_jtalk_dic_utf_8-1.11")
core = VoicevoxCore(open_jtalk_dict_dir=open_jtalk_dict_dir)

is_start = False  # 測定開始フラグ
is_end = False  # 測定終了フラグ
is_saved = False  # 音声ファイル保存フラグ

if not core.is_model_loaded(SPREAKER_ID):
    core.load_model(SPREAKER_ID)


def chat(messages):
    data = {"model": MODEL, "messages": messages, "stream": True}
    r = requests.post(
        URL,
        json=data,
        stream=True,
    )
    r.raise_for_status()
    output = ""

    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
            # the response streams one token at a time, print that as we receive it
            print(content, end="", flush=True)

        if body.get("done", False):
            message["content"] = output
            wave_bytes = core.tts(output, SPREAKER_ID)

            # バイナリーデータをバイトストリームとして読み込む
            audio_stream = io.BytesIO(wave_bytes)
            # バイトストリームを再生可能なオブジェクトに変換
            wave_read = wave.open(audio_stream, "rb")
            wave_obj = sa.WaveObject.from_wave_read(wave_read)

            # 再生
            play_obj = wave_obj.play()
            play_obj.wait_done()
            return message


def sampling_voice():
    global is_start, is_end, is_saved
    CHUNK = 2**10
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    record_time = 0.5
    OUTPUT_PATH = "./output.wav"

    while True:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        frames = []
        while not is_start:
            time.sleep(0.2)
            # print("continue....")
            continue

        print("Start to record")
        while not is_end:
            for i in range(0, int(RATE / CHUNK * record_time)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                # print(data)
                frames.append(data)

        print("Stop to record")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(OUTPUT_PATH, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        wf.close()
        is_saved = True


def detect_key():
    global is_start, is_end
    while True:
        c = readchar.readkey()
        if c == "s":
            is_start = True
            print("Start!")
        if c == "q":
            is_end = True
            print("End!")


def main():
    global is_start, is_end, is_saved
    messages = []
    while True:
        if is_saved == False:
            continue
        is_start = False
        is_end = False
        is_saved = False
        segments, info = model.transcribe(
            "output.wav",
            language="ja",
            chunk_length=15,
            condition_on_previous_text=False,
        )
        user_input = "30字以内で答えてください。"
        for segment in segments:
            user_input += segment.text
        print(user_input)

        messages.append({"role": "user", "content": user_input})
        message = chat(messages)
        messages.append(message)
        print("\n\n")
        print(messages)
        time.sleep(0.1)


if __name__ == "__main__":
    # スレッドを作る
    thread1 = threading.Thread(target=sampling_voice)
    thread2 = threading.Thread(target=detect_key)
    thread3 = threading.Thread(target=main)

    print("Press s to start")
    print("Press q to end")

    # スレッドの処理を開始
    thread1.start()
    thread2.start()
    thread3.start()

    main()

