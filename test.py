#!/usr/bin/env python3
"""
Lip Reading Model Test Script
Combines functionality from 3 original scripts to test the model via command line
"""

import os
import sys
import argparse
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from typing import List
import numpy as np

# Vocabulary and character mappings
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_model() -> Sequential:
    """Load the pre-trained lip reading model"""
    print("Loading model architecture...")
    model = Sequential()
    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))
    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))
    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))
    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))
    
    # Load weights - Updated paths for your folder structure
    weights_path = os.path.join('models - checkpoint 96', 'checkpoint')
    if not os.path.exists(weights_path):
        # Try alternative paths based on your folder structure
        alternative_paths = [
            r'D:\LipNet\models - checkpoint 96\checkpoint',
            'models - checkpoint 96/checkpoint',
            './models - checkpoint 96/checkpoint',
            'models/checkpoint', 
            r'D:\LipNet\models\checkpoint',
            './models/checkpoint',
            '../models/checkpoint',
            './checkpoint',
            'checkpoint'
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                weights_path = alt_path
                break
        else:
            print(f"Error: Model weights not found. Tried:")
            for alt_path in alternative_paths:
                print(f"  - {alt_path}")
            sys.exit(1)
    
    print(f"Loading weights from: {weights_path}")
    model.load_weights(weights_path)
    print("Model loaded successfully!")
    return model

def load_video(path: str) -> tf.Tensor:
    """Load and preprocess video frames"""
    print(f"Loading video: {path}")
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {path}")
    
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")
    
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        frame = tf.image.rgb_to_grayscale(frame)
        # Crop to lip region (adjust coordinates as needed)
        frames.append(frame[190:236, 80:220, :])
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{frame_count} frames")
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("No frames could be extracted from the video")
    
    # Normalize frames
    frames = tf.stack(frames)
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    normalized_frames = tf.cast((frames - mean), tf.float32) / std
    
    print(f"Video preprocessing complete. Shape: {normalized_frames.shape}")
    return normalized_frames

def load_alignments(path: str) -> tf.Tensor:
    """Load alignment data (ground truth) if available"""
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        
        tokens = []
        for line in lines:
            line = line.split()
            if len(line) > 2 and line[2] != 'sil':
                tokens.extend([' ', line[2]])
        
        return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]
    except FileNotFoundError:
        print(f"Warning: Alignment file not found: {path}")
        return None

def predict_from_video(model: Sequential, video_path: str) -> str:
    """Run prediction on a video file"""
    print(f"\n{'='*50}")
    print(f"PROCESSING VIDEO: {video_path}")
    print(f"{'='*50}")
    
    # Load and preprocess video
    frames = load_video(video_path)
    
    # Ensure we have exactly 75 frames (pad or truncate if necessary)
    if frames.shape[0] != 75:
        print(f"Adjusting frame count from {frames.shape[0]} to 75...")
        if frames.shape[0] > 75:
            frames = frames[:75]  # Truncate
        else:
            # Pad with last frame
            padding_needed = 75 - frames.shape[0]
            last_frame = frames[-1:] 
            padding = tf.repeat(last_frame, padding_needed, axis=0)
            frames = tf.concat([frames, padding], axis=0)
    
    # Add batch dimension
    video_input = tf.expand_dims(frames, axis=0)
    print(f"Model input shape: {video_input.shape}")
    
    # Make prediction
    print("Running prediction...")
    yhat = model.predict(video_input, verbose=1)
    print(f"Raw prediction shape: {yhat.shape}")
    
    # Decode prediction using CTC
    print("Decoding prediction...")
    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    print(f"Decoded tokens: {decoder}")
    
    # Convert to text
    converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
    
    return converted_prediction

def test_model(video_path: str = None, data_dir: str = None):
    """Test the model on video(s)"""
    # Load model
    model = load_model()
    
    if video_path:
        # Test single video
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
        
        try:
            prediction = predict_from_video(model, video_path)
            print(f"\n{'='*50}")
            print(f"PREDICTION RESULT:")
            print(f"{'='*50}")
            print(f"Video: {os.path.basename(video_path)}")
            print(f"Predicted Text: '{prediction}'")
            print(f"{'='*50}")
            
        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            
    elif data_dir:
        # Test multiple videos from directory
        if not os.path.exists(data_dir):
            print(f"Error: Data directory not found: {data_dir}")
            return
            
        video_extensions = ['.mp4', '.avi', '.mpg', '.mov', '.mkv']
        video_files = [f for f in os.listdir(data_dir) 
                      if any(f.lower().endswith(ext) for ext in video_extensions)]
        
        if not video_files:
            print(f"No video files found in directory: {data_dir}")
            return
            
        print(f"Found {len(video_files)} video files")
        
        results = []
        for i, video_file in enumerate(video_files, 1):
            video_path = os.path.join(data_dir, video_file)
            print(f"\n[{i}/{len(video_files)}] Processing: {video_file}")
            
            try:
                prediction = predict_from_video(model, video_path)
                results.append((video_file, prediction))
                print(f"✓ Completed: {video_file} -> '{prediction}'")
            except Exception as e:
                print(f"✗ Error processing {video_file}: {str(e)}")
                results.append((video_file, f"ERROR: {str(e)}"))
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS SUMMARY")
        print(f"{'='*60}")
        for video_file, prediction in results:
            print(f"{video_file:30} -> {prediction}")
        print(f"{'='*60}")
    
    else:
        # Default: try to find videos in standard location
        default_paths = [
            r'D:\LipNet\data\s1',
            os.path.join('data', 's1'),
            os.path.join('..', 'data', 's1'),
            './data/s1'
        ]
        
        for default_path in default_paths:
            if os.path.exists(default_path):
                print(f"No specific path provided. Testing videos in: {default_path}")
                test_model(data_dir=default_path)
                return
        
        print("Error: No video path provided and default data directory not found.")
        print("Tried looking in:")
        for path in default_paths:
            print(f"  - {path}")
        print("\nUsage:")
        print("  python test_model.py --video path/to/video.mp4")
        print("  python test_model.py --data-dir path/to/video/directory")
        print(f"  python test_model.py --data-dir \"D:\\LipNet\\data\\s1\"")

def main():
    parser = argparse.ArgumentParser(description='Test Lip Reading Model')
    parser.add_argument('--video', '-v', type=str, help='Path to single video file')
    parser.add_argument('--data-dir', '-d', type=str, help='Path to directory containing videos')
    parser.add_argument('--list-videos', '-l', action='store_true', help='List available videos in default directory')
    
    args = parser.parse_args()
    
    if args.list_videos:
        default_paths = [
            r'D:\LipNet\data\s1',
            os.path.join('data', 's1'),
            os.path.join('..', 'data', 's1'),
            './data/s1'
        ]
        
        for default_path in default_paths:
            if os.path.exists(default_path):
                videos = [f for f in os.listdir(default_path) if f.endswith(('.mp4', '.avi', '.mpg', '.mov', '.mkv'))]
                print(f"Available videos in {default_path}:")
                for i, video in enumerate(videos, 1):
                    print(f"  {i}. {video}")
                return
        
        print("Default directory not found in any of these locations:")
        for path in default_paths:
            print(f"  - {path}")
        return
    
    try:
        test_model(args.video, args.data_dir)
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()