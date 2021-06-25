import flash
import argparse
from flash.image import ImageClassificationData, ImageClassifier

"python src/train_model.py --dataset ik --data_root data/model_data"
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train an image classification model')
  parser.add_argument('--dataset', type=str, 
                      help='dataset name')
  parser.add_argument('--data_root', type=str, 
                      help='path to data_root')

  args = parser.parse_args()

  # 2. Load the data
  datamodule = ImageClassificationData.from_folders(
      train_folder=f"{args.data_root}/train",
      val_folder=f"{args.data_root}/valid/",
      test_folder=f"{args.data_root}/valid/",
  )

  # 3. Build the model
  model = ImageClassifier(num_classes=datamodule.num_classes, backbone="resnet18")

  # 4. Create the trainer. Run once on data
  trainer = flash.Trainer(max_epochs=50)

  # 5. Finetune the model
  trainer.finetune(model, datamodule=datamodule, strategy="freeze")

  # 6. Save it!
  trainer.save_checkpoint("image_classification_model.pt")