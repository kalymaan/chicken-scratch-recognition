# chicken-scratch-recognition

Training a model that can recognize my beautiful handwriting.




### Software Requirements:
    - Pandas
    - Numpy 
    - Matplotlib  
    - Streamlit
    - TensorFlow 
    
### Application:

To run the application:

   - Install all required libraries listed above under 'Software Requirements'
   - Download data/words.tgz [from here](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database)
       - unpack tgz file and place in data/ directory in the following structure:
       - chicken-scratch-recognition(home repo)
           - data
               - IAM_Words
                   - words.txt
                   - words
                       - a01 - r06
   - Navigate to **/streamlit_app** directory and run the following console command
```console
streamlit run streamlit_app.py
```

To use application. Input png files of cropped, handwritten words

Here is what the app should look like:

<img src =media/streamlit_image.PNG/>


### File structure:

- Code folder
    - [word_recognition.ipynb](code/word_recognition.ipynb)
        - A thorough walkthrough of the entire process.
        - Includes:
            - Exploration of Data with relevant visualizations
            - Thorough preprocessing techniques 
            - Explanation of CTC and model training
        - Requirements (listed under 'Software Requirements' above)
    - [word_recognizer_model_only.ipynb](code/word_recognizer_model_only.ipynb)
        - Solely used for time intensive model training. 
    - [true_inference.ipynb](code/true_inference.ipynb)
        - Used to create modular functions for eventual use in streamlit app    
- Data folder (not in repo)
    - Further explanation down below under 'Data Description'
        
- Media folder
    - Images used/created throughout the project

- Models_Pickles folder
    - **prediction_model_2.keras** - Final model that is used in the Streamlit App
    - Various other models and pickles used/generated in project
 

- Streamlit folder
    - **streamlit_app_v1.py** - streamlit app


    
### Data Description:

Over 96,456 handwritten words from the IAM dataset. Commonly used in OCR applications

Data Acquisition: data/words.tgz - Contains words (example: a01/a01-122/a01-122-s01-02.png).[Downloaded From Here](https://fki.tic.heia-fr.ch/databases/download-the-iam-handwriting-database)

Data Ingestion and Cleaning: Thorough process is outlined in [word_recognition.ipynb](code/word_recognition.ipynb) 

data/IAM_Words/words.text - is a pseudo data dictionary. The following is a dataframe with th pertinent information extracted from words.txt

<img src = 'media/words_tx_df.PNG'>

##### Columns:
    - word_id - a unique identifier for each png. corresponds to filename

    - seg_result - result of word segmentation. ok and err
    - gray_level - graylevel to binarize the line containing this word
    - x - -> bounding box around this word in x,y,w,h format
    - y - -> bounding box around this word in x,y,w,h format
    - w - -> bounding box around this word in x,y,w,h format
    - h - -> bounding box around this word in x,y,w,h format
    - grammatical_tag - the grammatical tag for this word
    - word -  the transcription for this word


### Contributor:
    
**Kalyan Lakshmanan** 
    - [Github](https://github.com/kalymaan) 
    - [LinkedIn](https://www.linkedin.com/in/kalyanlakshmanan/) 


