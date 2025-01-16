import pyaudio
import wave
import whisper

import time    # 時間計測用

model = whisper.load_model("tiny")


def main():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    while True:
        try:
            # Record
            d = stream.read(CHUNK)
            frames.append(d)

        except KeyboardInterrupt:
            # Ctrl - c
            break

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()


def recognize():    # decodeを使用

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("output.wav")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    #_, probs = model.detect_language(mel)
    #print(f"Detected language: {max(probs, key=probs.get)}")
    
    # decode the audio
    options = whisper.DecodingOptions(language="ja", fp16=False)
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)


def recognize2():    # transcribeを使用

    result = model.transcribe(
        "output.wav",
        verbose=True,
        language="ja",
        fp16=False
    )
    print(result["text"])


if __name__ == '__main__':
    main()

    '''
    t1 = time.perf_counter()
    recognize()
    print(time.perf_counter() - t1)
    '''

    t2 = time.perf_counter()
    recognize2()
    print(time.perf_counter() - t2)


"""
メモ
・CPUのときはfp16=Falseにしないと警告が出るためつけている
・decodeメソッドはtranscribeに比べて低レベルらしい
　・tinyモデルでは基本的にtranscribeの方が速そう

・精度を上げたい場合
　・fine-tuningをする
　・promptを使用する
　　・transcribeの引数に「initial_prompt='ヒント'」
　　・少し試したがあまりうまくいかない
"""