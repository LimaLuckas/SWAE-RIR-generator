#!/usr/bin/env python3

import sys
import os
import pickle
import numpy as np
from scipy import signal
from scipy import stats
import librosa
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
import torchaudio
import torch
import json


class RIRData :
  
  def __init__(self, dataFilePath):
 
   self.dataFilePath  = dataFilePath
   self.sampling_rate=44100 # 44100 sample points per second
   self.reduced_sampling_rate=22050 # 22050 sample points per second
   self.rir_seconds=2
   self.track_length=self.rir_seconds*self.sampling_rate 
   self.final_sound_data_length=int(self.track_length/self.rir_seconds)
   self.roomProperties={}
   self.rooms_and_configs={}
   #self.rir_data_file_path=self.data_dir+"/RIR.pickle.dat"
   
   self.rir_data=[]  ##  "RIR.dat" --> list of list [34]
   if  os.path.exists(self.dataFilePath) :
         rir_data_file=open(self.dataFilePath,'rb')
         self.rir_data=pickle.load(rir_data_file)
         rir_data_file.close()
   else :
         print("Data file "+ self.dataFilePath + " not found, exiting ...")
         exit(1)

   print("rirData Length ="+str(len(self.rir_data)))
   self.rir_data_field_numbers={"timestamp":0,"speakerMotorIterationNo":1,"microphoneMotorIterationNo":2,"speakerMotorIterationDirection":3,"currentActiveSpeakerNo":4,"currentActiveSpeakerChannelNo":5,
                                "physicalSpeakerNo":6,"microphoneStandInitialCoordinateX":7,"microphoneStandInitialCoordinateY":8,"microphoneStandInitialCoordinateZ":9,"speakerStandInitialCoordinateX":10,
                                "speakerStandInitialCoordinateY":11,"speakerStandInitialCoordinateZ":12,"microphoneMotorPosition":13,"speakerMotorPosition":14,"temperatureAtMicrohponeStand":15,
                                "humidityAtMicrohponeStand":16,"temperatureAtMSpeakerStand":17,"humidityAtSpeakerStand":18,"tempHumTimestamp":19,"speakerRelativeCoordinateX":20,"speakerRelativeCoordinateY":21,
                                "speakerRelativeCoordinateZ":22,"microphoneStandAngle":23,"speakerStandAngle":24,"speakerAngleTheta":25,"speakerAnglePhi":26,"mic_RelativeCoordinateX":27,"mic_RelativeCoordinateY":28,
                                "mic_RelativeCoordinateZ":29,"mic_DirectionX":30,"mic_DirectionY":31,"mic_DirectionZ":32,"mic_Theta":33,"mic_Phi":34,"essFilePath":35,
                                "roomId":36,"configId":37,"micNo":38, 
                                "roomWidth":39,"roomHeight":40,"roomDepth":41, 
                                "rt60":42, 
                                "rirData":43 
                              } 
                             
   ## essFilePath =   <room_id> / <config_id> / <spkstep-SPKSTEPNO-micstep-MICSTEPNO-spkno-SPKNO> / receivedEssSignal-MICNO.wav
   
                                 
   # micNo
   #
   #    5          1
   #    |          |
   # 4-------||-------0
   #    |    ||    |
   #    6    ||    2                    


   # physicalSpeakerNo
   #              
   # 3---2---||
   #         || \    
   #         ||   1                      
   #         ||     \                      
   #         ||      0                    

  def getMetadata(self):
            rir_metadata = []
            
            for dataline in self.rir_data:
              
                CENT=100 ## M / CM 
            
                roomDepth=float(dataline[int(self.rir_data_field_numbers['roomDepth'])])/CENT # CM to M
                roomWidth=float(dataline[int(self.rir_data_field_numbers['roomWidth'])])/CENT # CM to M
                roomHeight=float(dataline[int(self.rir_data_field_numbers['roomHeight'])])/CENT # CM to M
                    
                microphoneCoordinatesX=float(dataline[int(self.rir_data_field_numbers['microphoneStandInitialCoordinateX'])])/CENT +float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateX'])])/CENT # CM to M 
                microphoneCoordinatesY=float(dataline[int(self.rir_data_field_numbers['microphoneStandInitialCoordinateY'])])/CENT +float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateY'])])/CENT # CM to M
                microphoneCoordinatesZ=float(dataline[int(self.rir_data_field_numbers['mic_RelativeCoordinateZ'])])/CENT


                speakerCoordinatesX=float(dataline[int(self.rir_data_field_numbers['speakerStandInitialCoordinateX'])])/CENT +float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateX'])])/CENT # CM to M
                speakerCoordinatesY=float(dataline[int(self.rir_data_field_numbers['speakerStandInitialCoordinateY'])])/CENT +float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateY'])])/CENT # CM to M
                speakerCoordinatesZ=float(dataline[int(self.rir_data_field_numbers['speakerRelativeCoordinateZ'])])/CENT

                rt60=float(dataline[int(self.rir_data_field_numbers['rt60'])])

                if 0.5 < rt60 < 0.6 :
                  rt60=rt60+0.1
                elif 0.6 < rt60:
                  rt60=rt60+0.2
                      
                speakerMotorIterationNo=int(dataline[int(self.rir_data_field_numbers['speakerMotorIterationNo'])])
                microphoneMotorIterationNo=int(dataline[int(self.rir_data_field_numbers['microphoneMotorIterationNo'])])
                currentActiveSpeakerNo=int(dataline[int(self.rir_data_field_numbers['currentActiveSpeakerNo'])])
                currentActiveSpeakerChannelNo=int(dataline[int(self.rir_data_field_numbers['currentActiveSpeakerChannelNo'])])
                physicalSpeakerNo=int(dataline[int(self.rir_data_field_numbers['physicalSpeakerNo'])]) 
                roomId=dataline[int(self.rir_data_field_numbers['roomId'])] 
                configId=dataline[int(self.rir_data_field_numbers['configId'])] 
                micNo=dataline[int(self.rir_data_field_numbers['micNo'])] 
                rirData=dataline[int(self.rir_data_field_numbers['rirData'])]
                lengthOfRirSignal=len(rirData)
                maxOfRirSignal=max(rirData)
                minOfRirSignal=min(rirData)

                rir_metadata.append({
                  "rt_60": float(rt60),
                  "mic_no": float(micNo),
                  "room_depth": float(roomDepth),
                  "room_width": float(roomWidth),
                  "room_height": float(roomHeight),
                  "mic_x_coord": float(microphoneCoordinatesX),
                  "mic_y_coord": float(microphoneCoordinatesY),
                  "mic_z_coord": float(microphoneCoordinatesZ),
                  "spk_x_coord": float(speakerCoordinatesX),
                  "spk_y_coord": float(speakerCoordinatesY),
                  "spk_z_coord": float(speakerCoordinatesZ),
                  "speaker_motor_iter": int(speakerMotorIterationNo),
                  "microphone_motor_iter": int(microphoneMotorIterationNo),
                  "current_active_speaker": int(currentActiveSpeakerNo),
                  "current_active_speaker_channel": int(currentActiveSpeakerChannelNo),
                  "physical_speaker_no": int(physicalSpeakerNo),
                })
                
            print("All records printed to all_records.txt")
            return rir_metadata
  
  def get_rir_data(self):
    rir_data = []
    
    for dataline in self.rir_data:
        # Extract the RIR data from the dataline
        rir = dataline[int(self.rir_data_field_numbers['rirData'])]
        # Convert the RIR data to a numpy array
        rir_array = np.array(rir)
        # Append the RIR data to the list
        rir_data.append(rir_array)
    # Convert the list of RIR data to a numpy array
    rir_data = np.array(rir_data)
    # Reshape the RIR data to have the correct dimensions
    rir_data = rir_data.reshape(len(rir_data), -1)
    # Return the RIR data
    return rir_data

  def get_mfcc(self, data):
    
    #Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Convert data to PyTorch tensor
    data_tensor = torch.tensor(data).to(device)
    # Compute MFCCs
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=self.sampling_rate,
        n_mfcc=25,
        melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128, "center": False}
    ).to(device)
    spec = mfcc(data_tensor)
    # Convert to numpy array
    mfcc = spec.cpu().numpy()
    return mfcc

  def normalize_mfcc(self, mfcc):
    # Normalize the MFCCs
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    return mfcc


def main():
    # Example usage
    dataFilePath = "/home/lucas/ML/gtu-rir/RIR.pickle.dat"
    rir_data = RIRData(dataFilePath)
    
    # Get RIR data
    rirs = rir_data.get_rir_data()
    
    # Get MFCCs
    mfcc = rir_data.get_mfcc(rirs)
    
    # Normalize MFCCs
    normalized_mfcc = rir_data.normalize_mfcc(mfcc)
    
    print("MFCCs:", normalized_mfcc)

if __name__ == "__main__":
    main()




