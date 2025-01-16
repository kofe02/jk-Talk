from openai import OpenAI
from pathlib import Path
from voicevox_core import VoicevoxCore
import simpleaudio as sa
import io
import json
import wave

talk = True
TALK_ON = "talk on"
TALK_OFF = "talk off"
SPEECH_SPEED = 1.5 #読み上げ速度

# コアファイルの設定
open_jtalk_dict_dir=Path("open_jtalk_dic_utf_8-1.11")
core = VoicevoxCore(open_jtalk_dict_dir=open_jtalk_dict_dir)

# ずんだもんの声ID
speaker_id = 3
# モデルデータの読み込み
if not core.is_model_loaded(speaker_id):
    core.load_model(speaker_id)

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama',
)

MODEL = "dsasai/llama3-elyza-jp-8b:latest"

# 保存するファイル名
history_file = "conversation_history.json"

# 履歴の読み込み
if Path(history_file).exists():
    with open(history_file, 'r') as file:
        messages = json.load(file)
else:
    messages = []

try:
    while True:
        role = "user"
        message = input("?")

        # Exit command
        if message.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Toggle Talk Mode
        if message.strip().lower() in (TALK_OFF.lower(), TALK_ON.lower()):
            talk = message.strip().lower() == TALK_ON.lower()
            print(f"トークモードは{'オン' if talk else 'オフ'}です")
            continue

        # Add user message to history
        messages.append({"role": role, "content": message})

        # Send to LLM
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages[-10:]  # Limit to last 10 messages
            )
        except Exception as e:
            print(f"API Error: {e}")
            continue

        llm_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": llm_response})

        # Text-to-Speech
        if talk:
            try:
                audio_query = core.audio_query(llm_response, speaker_id)
                audio_query.speed_scale = SPEECH_SPEED
                wave_bytes = core.synthesis(audio_query, speaker_id)

                # Play audio
                audio_stream = io.BytesIO(wave_bytes)
                wave_obj = sa.WaveObject.from_wave_read(wave.open(audio_stream, 'rb'))
                wave_obj.play()
            except Exception as e:
                print(f"TTS Error: {e}")

        # Save History
        try:
            with open(history_file, 'w') as file:
                json.dump(messages, file, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"File Save Error: {e}")

        print(llm_response)
except KeyboardInterrupt:
    print("\nGoodbye!")
