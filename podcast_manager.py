import random
from xml.etree.ElementTree import ElementTree

from pydub import AudioSegment
import xml.etree.ElementTree as ET
import re
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import os
import shutil
import time

"""
How to run

run these 3 commands once on your mack.  No need to do it again once done.

brew install ffmpeg
pip install pydub
pip install openai

Creatie a podcast:

put this file anywehre in your system
whenever you want to create a new podcast
1. create a folder name it whatever say mypodcast
2. inside mypodcast folder create a file named input.txt
3. paste your podcast content inside input.txt
4. rename the variale salt, with your  folder name  (e.g.mypodcast)

run this file.


"""
start_time = time.time()

# CHANGE THIS 
salt = "hashmap"  

load_dotenv()
openai_api_key = os.getenv("open_ai_key")
os.environ["OPENAI_API_KEY"] = openai_api_key
client = OpenAI()

input_file = "input.txt"
xml_file_name = "file.xml"
output_file_name = salt + ".mp3"
BASE_DIR = os.getcwd()
salt_path =  os.path.join(BASE_DIR, salt)
#  on each  fresh run delete root folder    
#  then create new, we don't want older data conflicting with new data
if not os.path.exists(salt_path):
    os.makedirs(salt_path)
    
DATA_DIR = os.path.join(salt_path, "data")
if os.path.exists(DATA_DIR):
    shutil.rmtree(DATA_DIR)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
input_file_path = os.path.join(salt_path, input_file)
xml_file_path = os.path.join(salt_path,xml_file_name)
message_audio_path = os.path.join(salt_path, "tts_output")
output_file_path = os.path.join(salt_path, output_file_name)
if os.path.exists(message_audio_path):
    shutil.rmtree(message_audio_path)
if not os.path.exists(message_audio_path):
    os.makedirs(message_audio_path)

voice_pool = ["shimmer", "fable", "nova", "echo", "alloy", "onyx"]
random.shuffle(voice_pool)

def generate_audio(text, speaker, filename, voice):
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename

def process_xml_dialogue(tree):
    root = tree.getroot()
    speaker_to_voice = {}
    next_voice = 0
    batches = []

    current_speaker = None
    current_lines = []
    current_ids = []
    current_chars = 0
    MAX_CHARS = 800

    for message in root.findall("message"):
        speaker = message.get("speaker")
        text = message.text.strip()
        msg_id = int(message.get("id"))

        if speaker not in speaker_to_voice:
            speaker_to_voice[speaker] = voice_pool[next_voice % len(voice_pool)]
            next_voice += 1

        # Flush batch if speaker changes or char limit exceeded
        if speaker != current_speaker or current_chars + len(text) > MAX_CHARS:
            if current_lines:
                batches.append((current_ids[0], current_speaker, "\n".join(current_lines)))
            current_speaker = speaker
            current_lines = [text]
            current_ids = [msg_id]
            current_chars = len(text)
        else:
            current_lines.append(text)
            current_ids.append(msg_id)
            current_chars += len(text)

    # Add last batch
    if current_lines:
        batches.append((current_ids[0], current_speaker, "\n".join(current_lines)))

    # Generate audio
    for idx, speaker, chunk_text in sorted(batches):
        voice = speaker_to_voice[speaker]
        filename = f"{message_audio_path}/{idx:03d}_{speaker}.mp3"
        print(f"Generating audio for {speaker} (chunk {idx}): {chunk_text[:40]}...")
        generate_audio(chunk_text, speaker, filename, voice)


def get_xml(text, lines_per_chunk=40):
    print("creating structured format.")
    lines = text.strip().splitlines()
    chunks = []

    for i in range(0, len(lines), lines_per_chunk):
        chunk_lines = lines[i:i+lines_per_chunk]
        prompt = f"""
        You are given a conversation as plain text, one line per message.

Your job is:
1. Identify or infer the speaker for each line.
2. Alternate logically between speaker A and B when not labeled.
3. Output the conversation in the following format:
   A ||| message here
   B ||| response here

Use the literal string `|||` (three pipe symbols) as the separator between speaker and message.

Do not add any extra commentary, tags, or formatting. Only return the structured conversation lines.

Input: {chr(10).join(chunk_lines)} 

Output:

"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        annotated = response.choices[0].message.content.strip()
        chunks.append(annotated)

    print("done.")
    return "\n".join(chunks)



def write_xml(input_data):
    print("creating xml")
    xml_data = input_data.strip()
    conversation = ET.Element("conversation")

    for idx, line in enumerate(xml_data.splitlines()):
        if "|||" not in line:
            print(f"Skipping malformed line: {repr(line)}")
            continue

        speaker, sentence = line.split("|||", 1)
        message = ET.Element("message",speaker=speaker.strip(),id=str(idx))
        message.text = sentence.strip()  # assign sequential ID
        conversation.append(message)

    tree = ElementTree(conversation)
    tree.write(xml_file_path, encoding="utf-8", xml_declaration=True)
    print("done")



def merge_audio_clips():
    print("starting podcast creation")
    audio_files = sorted(
        [f for f in os.listdir(message_audio_path) if f.endswith(".mp3")],
        key=lambda x: int(x.split("_")[0])  # sort by leading index
    )
    final_audio = AudioSegment.empty()
    for file in audio_files:
        clip = AudioSegment.from_file(os.path.join(message_audio_path, file))
        final_audio += clip + AudioSegment.silent(duration=400)  # 0.4 sec pause
    final_audio.export(output_file_path, format="mp3")
    print(f"Podcast created: {output_file_path}")



with open(input_file_path) as f:
    input_text = f.read()
xml_data = get_xml(input_text)
write_xml(xml_data)
tree = ET.parse(xml_file_path)
process_xml_dialogue(tree)
merge_audio_clips()

end_time = time.time()
elapsed = end_time - start_time
print(f"\n🎧 Podcast created in {elapsed:.2f} seconds.")
