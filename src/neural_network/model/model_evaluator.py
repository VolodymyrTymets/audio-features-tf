from src.logger.logger_service import Logger


class ModelEvaluator:
  def __init__(self):
    self.loger = Logger('ModelEvaluator')

  def evaluate(self, model, test_ds):
    if model is None:
      raise ValueError("Model not set")
    self.loger.log('Evaluation of model:', 'green')
    return model.evaluate(test_ds, return_dict=True)


