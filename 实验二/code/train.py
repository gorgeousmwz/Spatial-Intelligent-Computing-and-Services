# from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model
from ultralytics import YOLO


# experiment = Experiment(
#   api_key = "SVZN4FOR26HeiXN4uOSMmBfAE",
#   project_name = "yolov8_batch16",
#   workspace="gorgeousmwz"
# )

# # Report multiple hyperparameters using a dictionary:
# hyper_params = {
#    "learning_rate": 0.01,
#    "steps": 300,
#    "batch_size": 16,
# }
# experiment.log_parameters(hyper_params)


# Initialize and train your model
# Load a model
model = YOLO('/home/ntire23_2/mwz/ultralytics-main/weights/yolov8s.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='/home/ntire23_2/mwz/ultralytics-main/data/data.yaml', \
            epochs=300, imgsz=640,batch=24,workers=8,save_period=1,device=1)

# # Seamlessly log your Pytorch model
# log_model(experiment, model, model_name="yolov8_batch16")
