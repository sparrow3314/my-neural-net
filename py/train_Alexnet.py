from train_classic_nets import main


EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.004
MOMENTUM = 0.9
NUM_WORKERS = 8
LR_REDUCE_FACTOR = 0.5
LR_PATIENCE = 3
MIN_LEARNING_RATE = 1e-5
SAVE_EVERY = 0
WEIGHTS_NAME = "final_weights.pth"
SEED = None
DEVICE = "auto"


if __name__ == "__main__":
    main(
        "alexnet",
        default_epochs=EPOCHS,
        default_batch_size=BATCH_SIZE,
        default_lr=LEARNING_RATE,
        default_momentum=MOMENTUM,
        default_num_workers=NUM_WORKERS,
        default_lr_reduce_factor=LR_REDUCE_FACTOR,
        default_lr_patience=LR_PATIENCE,
        default_min_lr=MIN_LEARNING_RATE,
        default_save_every=SAVE_EVERY,
        default_weights_name=WEIGHTS_NAME,
        default_seed=SEED,
        default_device=DEVICE,
    )

# Final: 92.98%