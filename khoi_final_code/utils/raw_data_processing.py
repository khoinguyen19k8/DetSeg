#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:42:26 2021

@author: Martijn van Leeuwen
"""
from .Packages_file import *

class Data_processing():
     
    #%% Loading and saving section 
    def Save_dicom_as_nifti(self,Dicom_directory,Save_directory):
        """
        Description:
            Save dicom images to a specified directory

        Parameters
        ----------
        Dicom_directory : string
            Directory towards the dicom folder.
        Save_directory : string
            Directory towards the folder where you want to store the results.

        Returns
        -------
        None.

        """
        dicom2nifti.convert_directory(Dicom_directory, Save_directory,compression=False)
    
    
    def Save_image_data_as_nifti(self,save_path,image_names,Data,Header,Auto_name_gen=False,Segmentation=False): 
        """
        Desription:
           Save the numpy array as a nifti file.
        Parameters
        ----------
        save_path : TYPE
            Location of the folder where you want to store the final data (ending with a /).
        image_names : list of strings
            list or string containing the name(s) of the images. If data is a list of images, you can use ['Test_name'] and all of the images will be saved in the format:
                "Test_name_1.nii" with the number changing for all instances in the data. By changing the Test_name value, you can easily store all the images with a,
                likewise name. Note that if the user wants to use this, "Auto_name_gen" must be True
        Data : string
            Data that you want to save. 
        Auto_name_gen: Boolean
            Boolean value determining whether you want to automatically create the image_names as shown above. Default is False      
        Returns
        -------
        No output, but saves nifti file that are obtained from the loaded data "Loading_Dicom_data function"
        """
        if type(image_names)!=list:
            image_names=[image_names] 
        if Auto_name_gen:
           image_names=["%s_%d.nii"% (image_names[0],i) for i in range(len(Data))]
        nr=0
        for Image in Data:
            if Segmentation:
                Image = np.array(Image,dtype='uint8')  
            Nifti_image=nib.Nifti1Image(Image, affine=None, header=Header)
            Save_path=os.path.join(save_path,image_names[nr])
            nib.save(Nifti_image, Save_path)
            print(image_names[nr]+" saved")
            nr+=1

        return None
        
    def Loading_Nifti_data(self,path_nifti_folder,File_names=[""],Load_all=False,Resize=False,Resolution=(768,768),Mute=False,Binarize_label=False):
        """
        Description:
            This function can load in the nifti files that are stored in a certain folder. This function is needed when you have
            previously loaded the dicom images and saved them in a .nii format. This function loads the data much faster than repeatedly,
            loading DICOM data.

        Parameters
        ----------
        path_nifti_folder : string
            The directory to the files where the nifti folders are located.
        File_names : list of strings
            A list or string, containing the names of the nifti files in the folder defined in parameter:path_nifti_folder.
        Load_all : False or "All", optional
            Variable that defines wheter all the nii files in the folder should be loaded. Default value is "False". The default is False.
        Resize: Boolean
            This variable indicates whether the data must be resized.
        Resolution: Tuple (int,int)
            This varable indicates the desired dimensions of the slices when the user wants to resize the image.
        Mute: Boolean
            Determine whetehr you want to print statements during running of the code
        Binarize_label: Boolean
            Determine wheter you want to binarize the label. Default value is False
        Returns
        -------
        Nifti_files:TYPE
            The contents of the loaded nifti files.
        Headers:
            Headers of the loaded images
        """        

        All_files_in_folder=os.listdir(path_nifti_folder)
        Files_in_folder=[string for string in All_files_in_folder if string.find(".nii")!=-1] #Remove non patient files in directory
        Nifti_files=[]
        Headers=[]
        if Load_all==True:
            Files_to_load=Files_in_folder
        else:
            if type(File_names)==list:
                Files_to_load=File_names
            else:
                Files_to_load=[File_names]  
        if not Mute:        
            print("----Start Loading Files---")       
        for NF in tqdm(Files_to_load,disable=Mute):
            Load_path=os.path.join(path_nifti_folder,NF)
            Image_data=nib.load(Load_path)
            im_data=Image_data.get_fdata()
            if Resize and (im_data.shape[0],im_data.shape[1])!=Resolution:
                im_data=self.Resize_Patient_data(im_data, Resolution,Mute=Mute)
            if Binarize_label:
                Thresholded_lab=np.zeros(im_data.shape)
                Thresholded_lab[im_data>0]=1
                im_data=Thresholded_lab
                Thresholded_lab=[]
            Nifti_files.append(im_data)
            Headers.append(Image_data.header)
        if not Mute:
            print("---Nifti Data loaded Succesfully!---")
        

        return  Nifti_files,Headers
    

    
    def Loading_nrrd_data(self,data_path,File_names,Load_all=False,Resize=False,Resolution=(768,768)):
        """
        Description:
         -This function loads in the annotations that were done by the radiologists and stores the result in the class variable as a list
        of numpy arrays.           

        Parameters
        ----------
        data_path : string
            Path where all the patient folders are located.
        File_names : list or string containing the files with the annotation files.
            File name which the radiologists gave to the annotation file(s). Note that the annotations do not have an easily callable name so these need to be manyally 
            filled into a list.
        Load_all : False or "All", optional
            The default is False but can be changed to 'All' if you want to load the annotations of all of the patients.
        Resize: Boolean
            This variable indicates whether the data must be resized.
        Resolution: Tuple (int,int)
            This varable indicates the desired dimensions of the slices when the user wants to resize the image.
        Returns
        -------
        self.label_DATA: numpy.array
            A list containing the loaded nrrd annotations.

        """
        
        All_files_in_folder=os.listdir(data_path)
        Files_in_folder=[string for string in All_files_in_folder if string.find("P")!=-1] #Remove non patient files in directory    
        Label_data=[]
        if Load_all=="All":
            Files_to_load=Files_in_folder
        elif type (File_names)==list:
            Files_to_load=File_names
        else:
            Files_to_load=[File_names]
            
        for File in range(len(Files_to_load)):
            path = os.path.join(data_path,File_names[File])
            Label, header = nrrd.read(path)
            if Resize and (Label.shape[0],Label.shape[1])!=Resolution:
                Label=self.Resize_Patient_data(Label, Resolution)
            Label_data.append(Label)             
        print("Label(s) Loaded Sucessfully!")
        return Label_data


    def Save_nrrd_to_nii(self,Path_to_nrrd,Path_to_nii,File_names):
        """
        Description:
            In order to use the nrrd data and the dicom data, they must be converted to the same space. This function loads the nrrd data,
            alligns it with the dicom data, and saves it as a nifti file.

        Parameters
        ----------
        Path_to_nrrd : string
            Path to the folder where the nrrd data is stored.
        Path_to_nii : string
            path to where you want to store the nii data.
        File_names : string
            Name of the nrrd file.

        Returns
        -------
        None.

        """
        for file in File_names:
            nrrd=self.Loading_nrrd_data(Path_to_nrrd, [file])
            nrrd=np.flip(nrrd,axis=(2))
            nii_name=file[0:-5]+".nii"
            self.Save_image_data_as_nifti(Path_to_nii, [nii_name], nrrd, None)
        return None       
 
    
    def Normalise_Nifti_file(self,Data,Mute=False):
        """
        Description: 
            This function normalises the input data. This opperation is necessary when the data is stored in hounsonfield intensities.
            Before training the model, the data must be normalised. 

        Parameters
        ----------
        Data : List
            A list of numpy arrays that you want to normalise. 
        Returns
        -------
        Normalised_data: list
            List containing the normalised data.

        """
        if type(Data)!=list:
            Data=[Data]
        Normalised_data=[]
        if Mute==False:
            print("--- Start Normalising Data---")
        for i in tqdm(range(len(Data)),disable=Mute):
            Image_data=Data[i]
            Normalised_data.append((Image_data - np.min(Image_data))/(np.max(Image_data)-np.min(Image_data)))
        if Mute==False:
            print("---Data Normalised---")
        return Normalised_data      

    def Resize_Patient_data(self,Data,Resolution,Mute=False):
        """
        Description:
            This function can be used to Transform a specific volume to a volume where the slices have a specific size

        Parameters
        ----------
        Data : Numpy array
            Numpy array that contains the 3D CT volume.
        Resolution : (int,int)
            A tuple containing 2 values that define the desired size of the slice.  eg: (763,763)

        Returns
        -------
        data_resized : numpy array
            Numpy array with the specified dimension for the slices.

        """

        import cv2
        data_resized = []
        for i in range(0,Data.shape[2]):
            img_resized = cv2.resize(Data[:,:,i], dsize=Resolution, interpolation=cv2.INTER_AREA)
            data_resized.append(img_resized)
        data_resized = np.dstack(data_resized)
        if Mute==False:
            print("Data is resized, Original size was: (%s,%s,%s) and is now (%s,%s,%s)"%
              (Data.shape[0],Data.shape[1],Data.shape[2], data_resized.shape[0],data_resized.shape[1],data_resized.shape[2]))
        return data_resized
    

    
   
 
        
        
        
        