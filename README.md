
# Unveiling the Unseen: A Novel Video Identification Attack on Media Multiplexing
## Attack process of Anya

![image](https://github.com/user-attachments/assets/4644cceb-45d2-452e-a05f-29743ebd6943)




#### Step1: Fingerprint Collection
This process entails simulating client behavior and parsing segment information from webpage elements and the metadata of headers of audio and video files. The specific process is as follows:

- The `adaptiveFormats` field from the video webpage element must be accessed. This field provides the `start` and `end` values within the `indexRange` field, which indicate the byte range of the index box where segment information is recorded in the container header. For MP4 files, the index box is called `sidx`, while for WebM files, it is called `cues`.
 
- By locating the box in the video or audio file, four sequences can be parsed: the video segment size sequence, the video segment timeline sequence, the audio segment size sequence, and the audio segment timeline sequence.

#### Step2: Traffic Fingerprint Collection
The method involves extracting the chunk size from traffic and  correcting its value.

- Accumulating the downstream packets between two upstream packets.
 
- Then the size of the HTTP/3 and QUIC headers encapsulated in the packets are subtracted to obtain the actual size of the transmitted multiplexing chunk.

#### Step3: Attack Method
Our approach involves two steps: **fingerprint fusion** and **fuzzy search**. Fingerprint fusion simplifies the matching process, while fuzzy search eliminates the impact of uncertainties and removes dependency on sequence continuity. Additionally, we propose a strategy to enhance matching efficiency when dealing with large-scale fingerprint databases, further optimizing our approach's performance.

- Sorting the video segment and audio segment based on their respective timeline, and then align the end times of video segments with the start times of audio segments on the same timeline. This mapped time sequence is used to establish the order for fused fingerprints.

- The fused fingerprint needs to be converted into a prefix-sum sequence. Then Using scaling factors to obtain scaled prefixes and sequences and storing it in a hashmap. The implementation details of fuzzy search are shown in the following figure:
  
![image](https://github.com/user-attachments/assets/de41672a-6ede-4298-bf3f-6385fd6e0fc2)




## The code organization for each section is as follows:
- `src\extract_fingreprint` : fingerprint collection and processing

- `src\match` : attack method

- `data\fingerprint` : fingerprint file 

- `data\match_result` : example for match resullt

- `data\quic_chunk_body` : data used for fitting the liner regression model
