import argparse
import os
import sys
import yt_dlp
import re

def sanitize_filename(filename):
    """Convert filename to a safe version that works across different systems"""
    # Replace any non-alphanumeric characters (except for periods, hyphens and underscores) with underscores
    filename = re.sub(r'[^a-zA-Z0-9\.\-_]', '_', filename)
    return filename.strip()

def download_video(video_url):
    """
    Download video from YouTube link without audio
    Saves to the same directory as the script
    """
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Options for yt-dlp - get best video only (no audio)
        ydl_opts = {
            # Format selection for highest quality video without audio
            'format': 'bestvideo[ext=mp4]',
            'outtmpl': os.path.join(script_dir, '%(title)s.%(ext)s'),
            'restrictfilenames': True,  # Restrict filenames to ASCII chars
            'quiet': False,
            'no_warnings': False,
            'progress': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            if 'title' not in info:
                raise Exception("Failed to get video title")
            
            safe_title = sanitize_filename(info['title'])
            video_path = os.path.join(script_dir, f"{safe_title}.mp4")
            
            print(f"Downloading: {info['title']}")
            ydl_opts.update({'outtmpl': video_path})
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                ydl_download.download([video_url])
            
            print(f"Video downloaded to: {video_path}")
            return video_path
            
    except Exception as e:
        raise Exception(f"Error downloading video: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Download YouTube video without audio')
    parser.add_argument('video_url', help='YouTube video URL')
    args = parser.parse_args()
    
    try:
        download_video(args.video_url)
        print("Download completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()