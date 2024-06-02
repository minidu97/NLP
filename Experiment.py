import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "C:/Users/admin/Desktop/NLP/ffmpeg/bin/ffmpeg.exe"

import moviepy.editor as mp
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import torch
import time
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline


# Set the path to FFmpeg executable
ffmpeg_path = "C:/Users/admin/Desktop/NLP/ffmpeg/bin/ffmpeg.exe"

# Step 1: Extract audio from video
def extract_audio_from_video(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

# Step 2: Transcribe the audio using IBM Watson
def transcribe_audio_watson(audio_path, api_key, service_url):
    authenticator = IAMAuthenticator(api_key)
    speech_to_text = SpeechToTextV1(authenticator=authenticator)
    speech_to_text.set_service_url(service_url)

    with open(audio_path, 'rb') as audio_file:
        response = speech_to_text.recognize(
            audio=audio_file,
            content_type='audio/wav',
            model='en-US_BroadbandModel'
        ).get_result()

    transcript = " ".join(result['alternatives'][0]['transcript'] for result in response['results'])
    return transcript.strip()



def compare_transcriptions_with_evaluation(transcript1, transcript2, ground_truth):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Split transcripts into smaller segments
    segment_length = 512  # Adjust based on model's max length
    transcript1_segments = [transcript1[i:i+segment_length] for i in range(0, len(transcript1), segment_length)]
    transcript2_segments = [transcript2[i:i+segment_length] for i in range(0, len(transcript2), segment_length)]

    comparison_results = []
    total_matches = 0

    start_time = time.time()

    for index, (t1_seg, t2_seg, gt) in enumerate(zip(transcript1_segments, transcript2_segments, ground_truth)):
        input_text = f"Transcript 1: {t1_seg}\nTranscript 2: {t2_seg}\nAre there any conflicts between the two transcriptions?"
        
        # Tokenize the input text
        input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length)
        
        # Generate text
        output = model.generate(input_ids, max_length=1024, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)

        # Decode the generated text
        comparison = tokenizer.decode(output[0], skip_special_tokens=True)
        comparison_results.append(comparison)

        # Evaluate the result against the ground truth
        if comparison.strip() == gt.strip():
            total_matches += 1

    end_time = time.time()
    processing_time = end_time - start_time

    accuracy = total_matches / len(ground_truth)
    
    return {
        "comparison_results": comparison_results,
        "accuracy": accuracy,
        "processing_time": processing_time
    }

# Main function to run the process
def main(video_path, api_key, service_url):
    audio_path = "extracted_audio.wav"

    # Extract audio from video
    extract_audio_from_video(video_path, audio_path)

    # Transcribe audio using IBM Watson
    transcript1 = transcribe_audio_watson(audio_path, api_key, service_url)
    print("Transcript from IBM Watson:", transcript1)

    # For simplicity, let's assume transcript2 is another transcription method's output
    transcript2 = transcript1  # Replace with actual different method if available

    # Define ground truth comparisons for evaluation
    ground_truth = [
        "Expected comparison for segment 1",
        "Expected comparison for segment 2",
        # Add more ground truth comparisons for each segment
    ]

    # Compare transcriptions using GPT-2 and evaluate
    results = compare_transcriptions_with_evaluation(transcript1, transcript2, ground_truth)
    print("Comparison Results:", results["comparison_results"])
    print("Accuracy:", results["accuracy"])
    print("Processing Time:", results["processing_time"], "seconds")

if __name__ == "__main__":
    video_path = "C:/Users/admin/Desktop/NLP/VCE Legal Studies_ Civil case scenario (part 1).mp4"
    api_key = "vp4lPW3YRZZGkrfkmanskAUsiWK-OThmVd962wJaLUVa"
    service_url = "https://api.au-syd.speech-to-text.watson.cloud.ibm.com/instances/ec88f8b9-a9b3-46cf-94e6-4abcce4cc768"
    main(video_path, api_key, service_url)