# Music Genre Classifier

A simple music genre classifier using a Convolutional Neural Network.
This project treats the music genre classification as computer vision problem by extracting frequency spectrum information, in this case- MFCC features, from the audio signals.

- Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC. They are derived from a type of cepstral representation of the audio clip (a nonlinear "spectrum-of-a-spectrum")- [Wikipedia](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum#:~:text=Mel%2Dfrequency%20cepstral%20coefficients%20(MFCCs,%2Da%2Dspectrum%22).)

The representation of the MFCCs are somewhat akin to images in their array shape, dimensions and even visualisation as seen below:

![MFCC spectrum](https://haythamfayek.com/assets/posts/post1/filter_banks_raw.jpg)
Image source: https://haythamfayek.com/

These "images" are then fed into the CNN model, which as of yet still requires a some hyperparameter tuning.

Asides from the music genre classification itself as an objective, this project also aims to create and display a complete package (contained within audiosep) of reusable blocks of code.

# Data

Data Source: [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) downloaded from Kaggle.

- The trained genres in this project and their respective numeric labels are as follows:
```json
{ 
  0: "Blues",
  1: "Classical",
  2: "Country",
  3: "Disco",
  4: "Hiphop",
  5: "Metal",
  6: "pop",
  7: "Reggae",
  8: "Rock"
}
```

Each genre contains hundred 30-second samples.

# Preprocessing and model
The preprocessing steps involved parsing each genre folder, splitting each sample by an ascertained number of samples (segments) and then generating a set number of MFCCs for each segment. This invariably helps to increase the amount of training data. The results are stored then in a .json file.

For new data (outside of this dataset), the input is split into 30 second samples followed by the above preprocessing with the exception that instead of being stored, they are passed directly into the model and predictions are made for each segment of each 30-second sample. The maximum occuring prediction is chosen as the song's genre.

The model (not yet tuned) was trained with the following architecture:
![Model Summary](https://drive.google.com/file/d/1cjn49GjIfAciMGvGgIdIh_FV3Xr9wJZk/view?usp=sharing)

After 30 epochs, a validation accuracy of 75% and loss (sparse categorical crossentropy) of 0.6 was attained. 

- Prediction:
As of yet, new predictions are made by calling the `audiosep-run` script and passing the music file for prediction as an argument `sys.argv[1]`.

The next steps of the project will be:
- [x] get working model and...
- [x] reusable package
- [ ] hyperparameter tuning 
- [ ] containerization
- [ ] deployment and continuous integration into training.


