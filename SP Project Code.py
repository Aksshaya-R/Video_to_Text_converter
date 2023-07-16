import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr  
import moviepy.editor as mp 
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

clip = mp.VideoFileClip(r"The Lion and The Rabbit.mp4")
print('Converting video transcripts into text ...')
clip.audio.write_audiofile(r"converted.wav")

file_path = "converted.wav"
sample,frequency = librosa.load(file_path,sr=None)
ft = np.abs(librosa.stft(sample))

print("\nSampling Rate : ",frequency)
print("No of samples",len(sample))

duration_of_audio= len(sample)/frequency


# time domain representation of the signal
plt.figure()
librosa.display.waveplot(sample, frequency)
plt.xlabel("Time (seconds) ")
plt.ylabel("Amplitude")
plt.title(" time domain representation")
plt.show()

# frequency domain representation of the audio signal
plt.figure()
librosa.display.waveplot(ft)
plt.ylabel("Frequency (Hz)")
plt.title("frequency domain representation")
plt.show()



num_seconds_video= duration_of_audio  # stores the video's no of seconds
print("The video is {} seconds".format(num_seconds_video))
l=list(range(0,int(num_seconds_video)+1,int(num_seconds_video)))  # divides the video into chunks 

diz={}   # empty dictionary which stores text extracted from each chunk

print('\n\nConverting video transcripts into text ...')
# converts each slice of video into text format
for i in range(len(l)-1):
    
    # ffmpeg_extract_subclip(filename, t1, t2, targetname) where t1 is the initial time and t2 is the end time ( in seconds)
    # [chunks over lap by 2 seconds]
    ffmpeg_extract_subclip("The Lion and The Rabbit.mp4", l[i]-2*(l[i]!=0), l[i+1], targetname="chunks/cut{}.mp4".format(i+1)) 
    clip = mp.VideoFileClip(r"chunks/cut{}.mp4".format(i+1)) # Importing the new audio file created using function VideoFileClip(filename)
    clip.audio.write_audiofile(r"converted/converted{}.wav".format(i+1)) # converting to Waveform audio file (WAV)
    r = sr.Recognizer() # creates an instance recogniser , which represents a collection of speech recognition settings and functionality.
    audio = sr.AudioFile("converted/converted{}.wav".format(i+1)) # sends the wav file as input
    
    # Use Google’s Cloud Speech-to-text API to extract the text from the audio file in format wav
    with audio as source:
      r.adjust_for_ambient_noise(source)  
      audio_file = r.record(source)
    result = r.recognize_google(audio_file) # returns a string
    print("\n\nRecognized Text:")
    print(result)
    
    diz['chunk{}'.format(i+1)] =result

  
l_chunks=[diz['chunk{}'.format(i+1)]for i in range(len(diz)) ]  # list which contains extracted texts from all the chucks
text='\n'.join(map(str,l_chunks))  # joining the texts with new line character


# writes all these in a text file
with open('recognized.txt',mode ='w') as file:  
   file.write("Recognized Speech:") 
   file.write("\n") 
   file.write(text) 
   print("Finally ready!")  
