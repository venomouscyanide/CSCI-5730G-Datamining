# Ref: https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html
# Ref: https://www.kaggle.com/luisblanche/pytorch-forecasting-temporalfusiontransformer

import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer, Baseline, TemporalFusionTransformer, QuantileLoss
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import OrdinalEncoder

from project.prep_and_train import CleaningAndTrain, DataFiles


def temporal_ft(enhanced_dataset, enhanced_test_dataset):
    enhanced_dataset['time_idx'] = (enhanced_dataset['date'].dt.date - enhanced_dataset['date'].dt.date.min()).dt.days

    prediction_steps = 16
    max_prediction_length = prediction_steps

    max_encoder_length = 60  # Go back  60 Days
    training_cutoff = enhanced_dataset["time_idx"].max() - max_prediction_length

    for cols_to_convert in ['family', 'locale', 'locale_name', 'description', 'state', 'city', 'type_y', 'store_nbr',
                            'cluster', 'type_x']:
        enhanced_dataset[cols_to_convert] = enhanced_dataset[cols_to_convert].astype(str).astype('category')

    enhanced_dataset['transactions'].fillna(value=enhanced_dataset['transactions'].mean(),
                                            inplace=True)

    training = TimeSeriesDataSet(
        enhanced_dataset[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="sales",
        group_ids=["store_nbr", "family"],
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["store_nbr",
                             'locale',
                             'locale_name',
                             'transferred',
                             'description',
                             "family",
                             "city",
                             "state",
                             "cluster",
                             "type_y"],
        time_varying_known_categoricals=["type_x", ],
        time_varying_known_reals=["time_idx", "onpromotion", 'days_from_payday', 'dcoilwtico', "earthquake_effect"
                                  ],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "sales",
            "transactions",
        ],
        target_normalizer=GroupNormalizer(
            groups=["store_nbr", "family"], transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(training, enhanced_dataset, predict=True, stop_randomization=True)

    # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    baseline_predictions = Baseline().predict(val_dataloader)
    print("Before training:", (actuals - baseline_predictions).abs().mean().item())

    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        max_epochs=1,  # TODO: CHANGEEE!
        gpus=0,
        weights_summary="top",
        gradient_clip_val=0.1,
        limit_train_batches=30,  # coment in for training, running valiation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

    trainer.fit(
        tft,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions = best_tft.predict(val_dataloader)
    print("After training", (actuals - predictions).abs().mean())

    for cols_to_convert in ['family', 'locale', 'locale_name', 'description', 'state', 'city', 'type_y', 'store_nbr',
                            'cluster', 'type_x']:
        enhanced_test_dataset[cols_to_convert] = enhanced_test_dataset[cols_to_convert].astype(str).astype('category')

    encoder_data = enhanced_dataset[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]
    enhanced_test_dataset['time_idx'] = (
            enhanced_test_dataset['date'].dt.date - enhanced_dataset['date'].dt.date.min()
    ).dt.days
    last_data = enhanced_dataset[
        enhanced_dataset['time_idx'].isin(
            [idx - prediction_steps for idx in enhanced_test_dataset['time_idx'].unique()])
    ]
    last_data['time_idx'] = last_data['time_idx'] + prediction_steps
    merge_a = enhanced_test_dataset[[col for col in enhanced_test_dataset.columns if 'sales' not in col]]
    merge_b = last_data[['time_idx', 'store_nbr', 'family', 'sales', 'transactions']]

    decoder_data = pd.merge(merge_a, merge_b, on=['time_idx', 'store_nbr', 'family', ], how="left")

    # combine encoder and decoder data
    new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)

    predictions, x = best_tft.predict(new_prediction_data, mode="raw", return_x=True)
    interpolated_data = best_tft.interpret_output(predictions, reduction="sum")

    return best_tft


if __name__ == '__main__':
    oe_locale = OrdinalEncoder(dtype=np.int64)
    oe_city = OrdinalEncoder(dtype=np.int64)
    oe_state = OrdinalEncoder(dtype=np.int64)
    oe_type_y = OrdinalEncoder(dtype=np.int64)

    base_path = 'store-sales-time-series-forecasting'

    train_data = pd.read_csv(os.path.join(base_path, DataFiles.TRAIN))
    test_data = pd.read_csv(os.path.join(base_path, DataFiles.TEST))

    # with oil data + misc data
    enhanced_train_dataset = CleaningAndTrain().base_cleaning(base_path, train_data, oe_locale, oe_city, oe_state,
                                                              oe_type_y, index_date=False, misc=True, oil=True,
                                                              oe=False,
                                                              drop_holidays=False)
    enhanced_test_dataset = CleaningAndTrain().base_cleaning(base_path, test_data, oe_locale, oe_city, oe_state,
                                                             oe_type_y, index_date=False, misc=True, oil=True,
                                                             oe=False, drop_holidays=False)

    best_ft = temporal_ft(enhanced_train_dataset, enhanced_test_dataset)
