# Music Genre Classifier

- Description: 
A simple music genre classifier using a Convolutional Neural Network.
This project treats the music genre classification as computer vision problem by extracting MFCC features from the audio signals.

Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC. They are derived from a type of cepstral representation of the audio clip (a nonlinear "spectrum-of-a-spectrum")- [Wikipedia](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum#:~:text=Mel%2Dfrequency%20cepstral%20coefficients%20(MFCCs,%2Da%2Dspectrum%22).)

The representation of the MFCCs are somewhat akin to images in their array shape, dimensions and even visualisation.
These "images" are then into the CNN model, which still requires a bit more hyperparameter tuning.

Asides from the music genre classification itself as an objective, this project also aims to create and display a complete package of reusable blocks of code.

- Data Source: [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

The trained genres in this project and their respective numeric labels are as follows:
`{0: "Blues",
  1: "Classical",
  2: "Country",
  3: "Disco",
  4: "Hiphop",
  5: "Metal",
  6: "pop",
  7: "Reggae",
  8: "Rock"
  }`

        

# Preprocessing


