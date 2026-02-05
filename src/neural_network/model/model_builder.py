import tensorflow as tf
from src.files import Files
from src.logger.logger_service import Logger
from src.neural_network.model.stategies.build_strategy.build_strategy_interface import IModelBuildStrategy

class ModelBuilder:
  def __init__(self, strategy: IModelBuildStrategy, target_path: str = None):
    self.model = None
    self.target_path = target_path
    self.strategy = strategy
    self.files = Files()
    self.loger = Logger('ModelBuilder')
    self.checkpoint_dir = self.files.join(target_path, 'checkpoints')
    self.checkpoint_path = self.files.join(self.checkpoint_dir, '{epoch:02d}.weights.h5')

  def build(self, input_shape, output_shape, train_ds):
    self.loger.log(f'Building model with input shape: {input_shape}...')
    self.model = self.strategy.build(input_shape, output_shape, train_ds)
    self.loger.log(f'Model built: {self.model.name}', 'green')
    self.model.summary()
    return self.model

  def train(self, train_ds, val_ds, epochs):
    if self.model is None:
      raise ValueError("Model not set")
    initial_epoch = 1

    if self.files.is_exist(self.checkpoint_dir):
      self.loger.log(f'Loading model from checkpoint: {self.checkpoint_dir}')
      checkpoints = self.files.get_only_files(self.checkpoint_dir)
      checkpoints = sorted(checkpoints, key=lambda name: int(name.split('.')[0]), reverse=True)
      latest_checkpoint = checkpoints[0]
      if latest_checkpoint is not None:
        initial_epoch = int(latest_checkpoint.split('.')[0])
        self.loger.log(f'Model loaded from checkpoint: {latest_checkpoint}', 'green')
        self.model.load_weights(self.files.join(self.checkpoint_dir, latest_checkpoint))

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=self.checkpoint_path,
      save_weights_only=True,
      monitor='val_accuracy',
      mode='max',
      save_best_only=True)

    self.loger.log(f'Training mode from epoch: {initial_epoch} to {epochs}...', 'blue')
    if epochs <= initial_epoch:
      return None
    history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, initial_epoch=initial_epoch, callbacks=[cp_callback])
    self.loger.log(f'Model trained: {self.model.name}', 'green')
    self.model.save_weights(self.checkpoint_path.format(epoch=epochs))
    return history





