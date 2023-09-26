#%%
from model import HEDUNet
import torch
import torch.optim as optim
import numpy as np
import optuna
from optuna.trial import TrialState
import mlflow
from mlflow.tracking import MlflowClient
from utils import *
import math
#%%
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#%%
def objective(trial):
    
    with mlflow.start_run(run_name="HED-UNET", experiment_id=0):
        NUM_EPOCH = 200
        if trial.number == 30:
            print("trial stopped")
            trial.study.stop()
        model = HEDUNet(input_channels=3, output_channels=1).to(DEVICE)
        criterion = auto_weight_BCE
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = optim.Adam(model.parameters(), lr = lr)
        scaler = torch.cuda.amp.GradScaler()
        EARLY_STOPPING_TOLERANCE = 30
        image_dir = "../Data/output/images/"
        mask_dir = "../Data/output/labels/"
        train_loader, val_loader = get_data(image_dir, mask_dir, trainBS = 8, testBS = 64)
        # Training of the model.
        min_loss = np.Inf
        NO_IMPROV = 0
        MODEL_LIST = []
        for epoch in range(NUM_EPOCH):
            model.train()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)

            model.eval()
            with torch.no_grad():
                test_loss, test_pred, test_img, test_lbl = validate(model, val_loader, criterion, DEVICE)
                dice = dice_coefficient(test_pred,test_lbl)
                f1_score = float(dice.detach().cpu().numpy())
            
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            trial.report(dice, epoch)
            if test_loss < min_loss:
                MODEL_LIST.append(model)
                NO_IMPROV = 0
                min_loss = test_loss
            else:
                NO_IMPROV += 1
            mlflow.log_metric("F1_Score", f1_score, step=epoch)
            mlflow.log_metric("Test_Loss", math.sqrt(test_loss), step=epoch)
            mlflow.log_metric("Train_Loss", train_loss, step=epoch)
            mlflow.log_params(trial.params)

            if NO_IMPROV == EARLY_STOPPING_TOLERANCE:
                print(f'Early Stopping at epoch {epoch}')
                mlflow.log_metric('Early Stopping',epoch)
                break
            else:
                continue
        mlflow.pytorch.log_model(MODEL_LIST[-1],f"trial{trial.number}_Model")
        
        plot_result(test_img, test_lbl, test_pred, trial)
        mlflow.log_artifacts(f"../result/trial{trial.number}","data")
    
    return math.sqrt(test_loss)
# %%
if __name__=="__main__":
    study = optuna.create_study(study_name="HED_UNET", direction="minimize")
    study.optimize(objective)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    #
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
# %%
