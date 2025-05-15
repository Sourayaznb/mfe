import os
from Dataloader import WholeImageDataset
from Model import Model

if __name__ == "__main__":
    # Path to the dataset
    path = '/media/radiology/ULB-LISA/manifest-1713182663002/Advanced-MRI-Breast-Lesions'

    # Training parameters
    batch_size = 1
    patch_size = (128, 128, 32)
    max_epochs = 55  # Total training epochs
    save_epoch_freq = 10

    # Define experiment name and directories
    experiment_name = f"SwinUNETR_batch_{batch_size}_patchsize_{patch_size[0]}_tverskyLoss_epochs_{max_epochs}"
    checkpoints_dir = "checkpoints"
    expr_dir = os.path.join(checkpoints_dir, experiment_name)

    # Initialize dataset and dataloaders
    dataset = WholeImageDataset(path=path, batch_size=batch_size, split_file="split_ids.json")
    train_loader, val_loader = dataset.get_loaders() #with Dataloader


    #train_loader, val_loader, test_loader, train_ids, val_ids, test_ids = dataset.get_loaders()

    # Initialize the model
    model = Model(
        expr_dir=expr_dir,
        img_size=patch_size,
        in_channels=1, #1 pour substraction phase 3 and 1
        out_channels=2,
        feature_size=48,
        max_epochs=max_epochs,
        save_epoch_freq=save_epoch_freq,
    )

    # Train the model
    model.train(train_loader, val_loader)



