import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence
import pandas as pd

# Function to calculate coherence between channels
def calculate_coherence(window_data, sfreq, winLen=64):
    num_channels = window_data.shape[1]
    coherence_matrix = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            _, coherence_spectrum = coherence(window_data[:, i], window_data[:, j], fs=sfreq, nperseg=winLen)
            coherence_matrix[i, j] = np.mean(coherence_spectrum)  # Store average coherence
            coherence_matrix[j, i] = coherence_matrix[i, j]  # Since coherence matrix is symmetric
    return coherence_matrix

rootPath=r"D:\XXX" # dataset Location
dataPath = rootPath + "\derivatives"
window_length = 0.4  # Length of each time window in seconds > To calculate Coherence matrix
window_overlap = 0.2  # Overlap length in seconds
segment_length = 10
segment_overlap = 0

participants=pd.read_csv(rootPath + "\participants.tsv", sep='\t')
images = []
labels = []

for i in range(1, 89):       # Loop for files
    
    if i < 10:
        zero = '0'
    else:
        zero = ''
    file_ = f'sub-0{zero}{i}_task-eyesclosed_eeg.set'  # Using f-string for easier string formatting
    subPath = f'sub-0{zero}{i}/eeg/{file_}'  # Use forward slashes instead of backslashes
    path_ = os.path.join(dataPath, subPath)
    print(path_)

    if os.path.exists(path_):  # Check if the file exists before loading
        # Load the EEG data
        raw = mne.io.read_raw_eeglab(path_, preload=True)  # Load all data into memory
        data_np = raw.get_data()
        channels = raw.ch_names
        sfreq = raw.info['sfreq']
        
        # SEGMENTATION #####################################################################
        n_samples_per_segment = int(segment_length * sfreq)
        n_samples_overlap_segment = int(segment_overlap * sfreq)
        segment_step_size = n_samples_per_segment - n_samples_overlap_segment
        total_samples = raw.n_times       
        print('--------------Processing file #' + str(i) + ' with length ' + str(total_samples) + ' (' + str(int(total_samples/sfreq)) + ' sec)---------------')
        segments = []
        start_idx = 0
        while start_idx + n_samples_per_segment <= total_samples:
            end_idx = start_idx + n_samples_per_segment
            segment = raw.get_data(start=start_idx, stop=end_idx)
            segments.append(segment)
            start_idx += segment_step_size

        segments_array = np.array(segments)
        # SEGMENTATION #####################################################################
        
        for s,mySegment in enumerate(segments):            
            # Split the data into time windows with overlap
            window_length_samples = int(window_length * sfreq)
            overlap_samples = int(window_overlap * sfreq)
            start_samples = 0
            end_samples = window_length_samples
            
            image=np.zeros(171)
            
            while end_samples <= mySegment.shape[1]:          #while end_samples <= data_np.shape[1]:
                window_data = mySegment[:, start_samples:end_samples].T
                
                # Calculate coherence between channels
                coherence_matrix = calculate_coherence(window_data, sfreq, winLen=window_length_samples/4)
                m=coherence_matrix.shape[0]
                r,c=np.triu_indices(m,1)
                a=coherence_matrix[r,c]
                image=np.column_stack((image,a)) 
                # Visualize the coherence matrix
    #            plt.figure()
    #            plt.imshow(coherence_matrix, cmap='coolwarm', interpolation='nearest')
    #            plt.colorbar()
    #            plt.title('Coherence Matrix')
    #            plt.xlabel('Channels')
    #            plt.ylabel('Channels')
    #            plt.show()
    
#                # Print coherence values
#                for i, channel in enumerate(channels):
#                    print(f"Coherence for channel {channel}: {coherence_matrix[i]}")
    
                # Increment window positions
                start_samples += window_length_samples - overlap_samples
                end_samples = start_samples + window_length_samples
            image=image[:,1:]
            images.append(image)
            labels.append(participants.iloc[i-1]['Group'])   # Label for the current image
            print(f'File #{i} > Segment #{s+1} > Connectogram [{participants.iloc[i-1].Group}]')
            plt.figure()
            plt.imshow(image, cmap='coolwarm', interpolation='nearest')
            plt.colorbar()
            plt.title('Connectogram')
            plt.xlabel('Window')
            plt.ylabel('Coherence Graph //Flattened')
            plt.show()
            #if(s==0) : break
            
    else:
        print(f"File {path_} does not exist.")
