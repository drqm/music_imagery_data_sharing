input_dir = 'C:/Users/au571303/Documents/projects/memory_music_iEEG/stimuli/manipulation_normalized/old/';
output_dir = 'C:/Users/au571303/Documents/projects/memory_music_iEEG/stimuli/manipulation_normalized/';

tones = dir([input_dir '*.wav']);

for t = 1:length(tones)
    [wav,fs] = audioread([input_dir tones(t).name]);
    wav = mean(wav,2);
    audiowrite([output_dir,tones(t).name],wav,fs);
end