from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from audiosep.model import convnet_model
from audiosep.data import load_data, save_mfcc, split_data
import joblib
import pickle

JSON_PATH_MAIN = "../raw_data/genre/data.json"

class Trainer(object):
    
    def __init__(self, json_path, **kwargs):
        self.json_path = json_path
        self.kwargs = kwargs
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
            split_data(dataset_path= self.json_path)
        self.input_shape = self.X_train.shape[1:]
        self.model = convnet_model(self.input_shape)
        self.genres = {0: "Blues",
                       1: "Classical",
                       2: "Country",
                       3: "Disco",
                       4: "Hiphop",
                       5: "Metal",
                       6: "pop",
                       7: "Reggae",
                       8: "Rock"
                      }
        
    def train(self, epochs= 30, batch_size= 32, verbose= 0):
        es = EarlyStopping(patience=10, restore_best_weights=True)

        self.model.fit(self.X_train, self.y_train,
                       epochs= epochs,
                       batch_size= batch_size,
                       validation_data= (self.X_val, self.y_val),
                       verbose= verbose,
                       callbacks= [es])
    def evaluate(self, test=False):
        if test:
            test_metrics = self.model.evaluate(self.X_test, self.y_test, verbose= 0)
            test_print = f"test accuracy: {test_metrics[1]}"
            print(test_print)
        train_metrics = self.model.evaluate(self.X_train, self.y_train, verbose= 0)
        val_metrics = self.model.evaluate(self.X_val, self.y_val, verbose= 0)
        
        train_print = f"train loss: {train_metrics[0]}, train accuracy: {train_metrics[1]}"
        val_print =  f"val loss: {val_metrics[0]}, val accuracy: {val_metrics[1]}"
        
        print(train_print)
        print(val_print)
        
    def save_model(self, model_name):
        """Save the model into a HD5 format"""
        model_path = '../models/'
        self.model.save(model_path + model_name)
        print(f"{model_name} saved locally at {model_path + model_name}")
        
    def predict_new_song(self, X):
    
        test = save_mfcc(X, train= False, num_segments= 10, verbose=False)

        predictions = []

        for segment in test:
            segment = segment[np.newaxis, ...]
            seg_pred = self.model.predict(segment)
            seg_pred = np.argmax(seg_pred, axis=1)[0]
            predictions.append(seg_pred)

        predictions = np.array(predictions)

        values, counts = np.unique(predictions, return_counts= True)
        index = np.argmax(counts)
        
        # labels gotten from data.json['mapping']
        print(f"Predicted genre: {self.genres.get(values[index])}")
        
        #return values[index]

if __name__ == "__main__":
    train = Trainer(JSON_PATH_MAIN)
    train.train(epochs= 2, verbose= 2)
    train.evaluate(test= True)
    train.save_model('genre_model')
        