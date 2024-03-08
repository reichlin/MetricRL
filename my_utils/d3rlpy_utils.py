import numpy as np
import torch
import d3rlpy


def get_d3rlpy_algo(algo_name, gamma, batch_size, device_flag, baseline_hyper, use_images=0, discrete=False):

    hidden_units = [256] * 3
    activation = 'relu'


    if use_images == 0:
        model_architecture = d3rlpy.models.VectorEncoderFactory(hidden_units=hidden_units,#[64, 64, 64, 64],
                                                                activation=activation,#'relu',
                                                                use_batch_norm=False,
                                                                dropout_rate=None)
    else:
        model_architecture = d3rlpy.models.PixelEncoderFactory(filters=[(64, 3, 1), (64, 3, 2), (64, 3, 2), (64, 3, 2)],
                                                               feature_size=64,
                                                               activation=activation,#'relu',
                                                               use_batch_norm=False,
                                                               dropout_rate=None)

    if algo_name == 'DDPG':
        algo_class = d3rlpy.algos.DDPG
        agent = d3rlpy.algos.DDPGConfig(actor_encoder_factory=model_architecture,
                                        critic_encoder_factory=model_architecture,
                                        batch_size=batch_size,
                                        gamma=gamma,
                                        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                        n_critics=baseline_hyper['n_critics']).create(device=device_flag)
    elif algo_name == 'BC':
        if discrete:
            algo_class = d3rlpy.algos.DiscreteBC
            agent = d3rlpy.algos.DiscreteBCConfig(batch_size=batch_size,
                                                  observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                                  encoder_factory=model_architecture).create(device=device_flag)
        else:
            algo_class = d3rlpy.algos.BC
            agent = d3rlpy.algos.BCConfig(batch_size=batch_size,
                                          observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                          policy_type='deterministic',
                                          encoder_factory=model_architecture).create(device=device_flag)
    elif algo_name == 'CQL':
        if discrete:
            algo_class = d3rlpy.algos.DiscreteCQL
            agent = d3rlpy.algos.DiscreteCQLConfig(batch_size=batch_size,
                                                   observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                                   encoder_factory=model_architecture).create(device=device_flag)
        else:
            algo_class = d3rlpy.algos.CQL
            agent = d3rlpy.algos.CQLConfig(actor_encoder_factory=model_architecture,
                                           critic_encoder_factory=model_architecture,
                                           batch_size=batch_size,
                                           n_action_samples=baseline_hyper['n_actions'],
                                           gamma=gamma,
                                           observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                           conservative_weight=baseline_hyper['conservative_weight']).create(device=device_flag)
    elif algo_name == 'BCQ':
        if discrete:
            algo_class = d3rlpy.algos.DiscreteBCQ
            agent = d3rlpy.algos.DiscreteBCQConfig(encoder_factory=model_architecture,
                                                   batch_size=batch_size,
                                                   gamma=gamma,
                                                   observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                                   n_critics=baseline_hyper['n_critics']).create(device=device_flag)
        else:
            algo_class = d3rlpy.algos.BCQ
            agent = d3rlpy.algos.BCQConfig(actor_encoder_factory=model_architecture,
                                           critic_encoder_factory=model_architecture,
                                           imitator_encoder_factory=model_architecture,
                                           batch_size=batch_size,
                                           n_action_samples=baseline_hyper['n_actions'],
                                           gamma=gamma,
                                           observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                           n_critics=baseline_hyper['n_critics']).create(device=device_flag)
    elif algo_name == 'BEAR':
        algo_class = d3rlpy.algos.BEAR
        agent = d3rlpy.algos.BEARConfig(actor_encoder_factory=model_architecture,
                                        critic_encoder_factory=model_architecture,
                                        imitator_encoder_factory=model_architecture,
                                        batch_size=batch_size,
                                        n_action_samples=baseline_hyper['n_actions'],
                                        gamma=gamma,
                                        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                        n_critics=baseline_hyper['n_critics']).create(device=device_flag)
    elif algo_name == 'AWAC':
        algo_class = d3rlpy.algos.AWAC
        agent = d3rlpy.algos.AWACConfig(actor_encoder_factory=model_architecture,
                                        critic_encoder_factory=model_architecture,
                                        batch_size=batch_size,
                                        n_action_samples=baseline_hyper['n_actions'],
                                        gamma=gamma,
                                        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                        n_critics=baseline_hyper['n_critics']).create(device=device_flag)
    elif algo_name == 'PLAS':
        algo_class = d3rlpy.algos.PLAS
        agent = d3rlpy.algos.PLASConfig(actor_encoder_factory=model_architecture,
                                        critic_encoder_factory=model_architecture,
                                        imitator_encoder_factory=model_architecture,
                                        batch_size=batch_size,
                                        n_critics=baseline_hyper['n_critics'],
                                        gamma=gamma,
                                        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                        ).create(device=device_flag)
    elif algo_name == 'IQL':
        algo_class = d3rlpy.algos.IQL
        agent = d3rlpy.algos.IQLConfig(actor_encoder_factory=model_architecture,
                                       critic_encoder_factory=model_architecture,
                                       value_encoder_factory=model_architecture,
                                       batch_size=batch_size,
                                       gamma=gamma,
                                       observation_scaler=d3rlpy.preprocessing.PixelObservationScaler() if use_images == 1 else None,
                                       n_critics=baseline_hyper['n_critics'],
                                       expectile=baseline_hyper['expectile']).create(device=device_flag)

    return algo_class, agent

