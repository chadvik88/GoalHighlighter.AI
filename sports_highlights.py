import cv2
import torch
import easyocr
import numpy as np
import librosa
import soundfile as sf
import os

#Loading the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

#OCR model loading
reader = easyocr.Reader(['en'])

# Constants
CONFIDENCE_THRESHOLD = 0.4  
highlight_objects = {'sports ball', 'ball'}  
AUDIO_THRESHOLD = 0.6  #loud crowd cheers threshold
HIGHLIGHT_DURATION = 18  #Clip duration in seconds

# Video file input
input_video = "match.mp4" #place your video file here

#goal and object detection using OCR
def detect_goal_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    goal_frames = []
    last_score = None
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        #YOLO detection
        results = model(frame)
        detected_classes = results.pandas().xyxy[0]['name'].tolist()
        #ball tracking
        if any(obj in highlight_objects for obj in detected_classes):
            #scoreboard text extraction
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_text = reader.readtext(gray_frame, detail=0)
            print(f"Detected text at frame {frame_count}: {detected_text}") 
            #number extraction
            numbers = [int(s) for s in detected_text if s.isdigit()]
            if len(numbers) >= 1:  #one number represents score
                score = tuple(numbers[:1])  #Take the first number
                if score != last_score:
                    last_score = score
                    goal_frames.append(frame_count)
        frame_count += 1
    cap.release()
    return goal_frames, fps

#Audio extraction
def extract_audio(video_path, output_audio):
    os.system(f"ffmpeg -i {video_path} -q:a 0 -map a {output_audio} -y")

#Cheer detection
def analyze_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    rms = librosa.feature.rms(y=y)[0]
    cheer_spikes = np.where(rms > AUDIO_THRESHOLD)[0]
    print("Detected audio spikes:", cheer_spikes) 
    return cheer_spikes

#Audio fluctuations result in goal detection (delay by 2s)
def confirm_goals(goal_frames, audio_spikes, fps):
    goal_timestamps = [frame / fps for frame in goal_frames]
    confirmed_goals = []
    for goal_time in goal_timestamps:
        if any(abs(goal_time - (spike / fps)) < 2 for spike in audio_spikes): 
            confirmed_goals.append(goal_time)
    return confirmed_goals

#highlight extraction using ffmpeg
def extract_highlights(video_path, goal_times):
    highlight_clips = []
    for i, goal_time in enumerate(goal_times):
        start_time = max(goal_time - 2, 0)
        end_time = start_time + HIGHLIGHT_DURATION
        clip_filename = f"highlight_{i+1}.mp4" 
        os.system(f"ffmpeg -i {video_path} -ss {start_time} -to {end_time} -c copy {clip_filename} -y")
        highlight_clips.append(clip_filename)
    return highlight_clips

#highlight compiler
def merge_clips(clips, output_video="highlights.mp4"):
    if not clips:
        print("No clips to merge")
        return
    with open("file_list.txt", "w") as f:
        for clip in clips:
            f.write(f"file '{clip}'\n")
    os.system(f"ffmpeg -f concat -safe 0 -i file_list.txt -c copy {output_video} -y")
    os.remove("file_list.txt")  
    print("Highlights video created successfully!")

# primary pipeline
print("\n--- STEP 1: DETECTING THE GOALS ---")
goal_frames, fps = detect_goal_frames(input_video)
if not goal_frames:
    print("No goals detected. Exiting.")
else:
    print("\n--- STEP 2: EXTRACTING AUDIO FROM VIDEO ---")
    extract_audio(input_video, "audio.wav")
    print("\n--- STEP 3: ANALYZING AUDIO FOR CROWD CHEERS ---")
    audio_spikes = analyze_audio("audio.wav")
    print("\n--- STEP 4: CONFIRMING GOALS BASED ON AUDIO SPIKES ---")
    confirmed_goals = confirm_goals(goal_frames, audio_spikes, fps)
    print(f"Confirmed goal times (in seconds): {confirmed_goals}")
    if not confirmed_goals:
        print("No confirmed goals. Exiting.")
    else:
        print("\n--- STEP 5: EXTRACTING HIGHLIGHT CLIPS ---")
        clips = extract_highlights(input_video, confirmed_goals)
        print("\n--- STEP 6: CREATING HIGHLIGHTS COMPILATION ---")
        merge_clips(clips)

