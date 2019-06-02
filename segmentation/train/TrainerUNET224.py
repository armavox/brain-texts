import os
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
np.random.seed(0)


class TrainerUNET224:
    def __init__(self,model: Model, train_loader, val_loader, checkpoint_path,
                 epochs=5, train_steps_per_epoch=100, val_steps_per_epoch=10, key_for_saving="val_dice_coef"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_steps_per_epoch = train_steps_per_epoch
        self.val_steps_per_epoch = val_steps_per_epoch

        self.checkpoint_path = checkpoint_path
        self.plots_path = os.path.join(self.checkpoint_path, "plots")
        self.key_for_saving = key_for_saving

        self.epochs = epochs

        self.__prepare()

    def __prepare(self):
        os.makedirs(self.plots_path, exist_ok=True)

    def train_model(self):
        track_checkpoints = ModelCheckpoint(os.path.join(self.checkpoint_path, "weights-{epoch:02d}.hdf5"),
                                            verbose=1, save_best_only=True, save_weights_only=True, mode="max",
                                            monitor=self.key_for_saving)
        H = self.model.fit_generator(generator=self.train_loader,
                                     steps_per_epoch=self.train_steps_per_epoch,
                                     validation_data=self.val_loader,
                                     validation_steps=self.val_steps_per_epoch,
                                     epochs=self.epochs, callbacks=[track_checkpoints], verbose=1)

        time_str = datetime.now().strftime('%Y-%m-%d%H-%M-%S')

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(1, self.epochs + 1), H.history["loss"], color='blue', label="train_loss")
        plt.plot(np.arange(1, self.epochs + 1), H.history["val_loss"], color='red', label="test_loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig( os.path.join(self.plots_path, "loss_%s.png" % time_str))
        plt.close()

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(1, self.epochs + 1), H.history[self.key_for_saving], color='blue', label=self.key_for_saving)
        plt.title("Metric")
        plt.xlabel("Epoch #")
        plt.ylabel(self.key_for_saving)
        plt.legend(loc="upper right")
        plt.savefig( os.path.join(self.plots_path, "metric_%s.png" % time_str))
        plt.close()
