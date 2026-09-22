# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Independent multi-source causal action mid-training recipe."""

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from cosmos_framework.callbacks.causal_action_stats import CausalActionStats
from cosmos_framework.callbacks.every_n_draw_sample import EveryNDrawSample
from cosmos_framework.callbacks.memory_trace import MemoryTrace
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.causal_action_mixture import CausalActionMixture
from cosmos_framework.data.generator.action.datasets.causal_action_sft_dataset import (
    get_causal_action_droid_sft_dataset,
)
from cosmos_framework.data.generator.joint_dataloader import PackingDataLoader, RankPartitionedDataLoader
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cfg = LazyDict(
    dict(
        defaults=[
            {"override /model": "mot_causal_action_fsdp"},
            {"override /data_train": None},
            {"override /data_val": None},
            {"override /optimizer": "adamw"},
            {"override /scheduler": "lambdalinear"},
            {"override /checkpoint": "s3"},
            {"override /callbacks": ["basic", "optimization", "job_monitor"]},
            {"override /ema": "power"},
            {"override /tokenizer": "wan2pt2_tokenizer"},
            {"override /sound_tokenizer": None},
            {"override /vlm_config": None},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="cosmos3",
            group="causal_action",
            name="action_causal_midtrain_edge",
            wandb_mode="disabled",
        ),
        model=dict(
            config=OmegaConf.merge(
                LazyDict(EDGE_MODEL_CONFIG, flags={"allow_objects": True}),
                dict(
                    max_action_dim=80,
                    causal_action_debug_noise_seed=None,
                    causal_training_strategy="teacher_forcing",
                    joint_attn_implementation="teacher_forcing",
                    teacher_forcing_dense_mode="grouped_tnd",
                    teacher_forcing_tnd_max_kv_tokens=131072,
                    teacher_forcing_block_size_min=1,
                    teacher_forcing_block_size_max=4,
                    teacher_forcing_history_blocks_min=8,
                    teacher_forcing_history_blocks_max=8,
                    max_num_tokens_after_packing=-1,
                    compile=dict(enabled=False),
                    ema=dict(enabled=False),
                    tokenizer=dict(encode_exact_durations=[5, 9, 17, 33]),
                    parallelism=dict(
                        data_parallel_shard_degree=-1,
                        fsdp_mixed_precision_enabled=True,
                    ),
                ),
            ),
        ),
        optimizer=dict(
            betas=[0.9, 0.99],
            eps=1e-08,
            fused=True,
            keys_to_select=[
                "moe_gen",
                "time_embedder",
                "vae2llm",
                "llm2vae",
                "state2llm",
                "state_modality_embed",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "k_norm_und_for_gen",
            ],
            lr=1e-05,
            lr_multipliers={},
            optimizer_type="AdamW",
            weight_decay=0.05,
        ),
        scheduler=dict(
            lr_scheduler_type="LambdaLinear",
            cycle_lengths=[100],
            f_max=[1.0],
            f_min=[1.0],
            f_start=[1.0],
            verbosity_interval=0,
            warm_up_steps=[0],
        ),
        trainer=dict(
            distributed_parallelism="fsdp",
            grad_accum_iter=1,
            logging_iter=1,
            max_iter=100,
            max_val_iter=None,
            run_validation=False,
            run_validation_on_start=False,
            save_zero_checkpoint=False,
            seed=42,
            timeout_period=999999999,
            validation_iter=100,
            compile_config=dict(recompile_limit=8, use_duck_shape=False),
            cudnn=dict(benchmark=True, deterministic=False),
            ddp=dict(broadcast_buffers=True, find_unused_parameters=False, static_graph=True),
            grad_scaler_args=dict(enabled=False),
            callbacks=dict(
                dataloader_speed=dict(every_n=100, save_s3=False, step_size=1),
                device_monitor=dict(
                    every_n=200,
                    log_memory_detail=True,
                    save_s3=False,
                    step_size=1,
                    upload_every_n_mul=5,
                ),
                grad_clip=dict(clip_norm=1.0, force_finite=True),
                heart_beat=dict(every_n=200, save_s3=False, step_size=1, update_interval_in_minute=20),
                iter_speed=dict(every_n=1, hit_thres=50, save_s3=False, save_s3_every_log_n=500),
                low_precision=dict(update_iter=1),
                manual_gc=dict(every_n=5, gc_level=1, warm_up=1),
                param_count=dict(save_s3=False),
                skip_nan_step=dict(max_consecutive_nan=100),
                training_stats=dict(log_freq=1),
                causal_action_stats=L(CausalActionStats)(),
                memory_trace=L(MemoryTrace)(include_worker_full_info=False),
                every_n_sample_reg=L(EveryNDrawSample)(
                    every_n=5000,
                    n_viz_sample=1,
                    guidance=[3.0],
                    num_sampling_step=20,
                    save_s3=False,
                    save_local=True,
                    do_x0_prediction=False,
                ),
            ),
        ),
        checkpoint=dict(
            broadcast_via_filesystem=False,
            dcp_async_mode_enabled=False,
            enable_gcs_patch_in_boto3=True,
            keys_not_to_resume=[],
            keys_to_skip_loading=[
                "net_ema.",
                "state2llm",
                "state_modality_embed",
                "action2llm",
                "llm2action",
                "action_modality_embed",
                "action_pos_embed",
            ],
            load_ema_to_reg=False,
            load_path="???",
            load_training_state=False,
            only_load_scheduler_state=False,
            save_iter=100,
            strict_resume=False,
            verbose=True,
            hf_export=dict(
                enabled=False,
                export_every_n=1,
                hf_repo_id=None,
                upload_to_object_store=dict(bucket="", credentials="", enabled=False),
            ),
            jit=dict(device="cuda", dtype="bfloat16", enabled=False, input_shape=None, strict=True),
            load_from_object_store=dict(bucket="", credentials="", enabled=False),
            save_to_object_store=dict(bucket="", credentials="", enabled=False),
        ),
        dataloader_train=L(PackingDataLoader)(
            audio_sample_rate=48000,
            dataset_name="causal_action_midtrain",
            max_samples_per_batch=None,
            max_sequence_length=16384,
            patch_spatial=2,
            sound_latent_fps=0,
            tokenizer_spatial_compression_factor=16,
            tokenizer_temporal_compression_factor=4,
            dataloader=L(RankPartitionedDataLoader)(
                batch_size=1,
                in_order=False,
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=2,
                sampler=None,
                datasets=dict(
                    robots=dict(
                        ratio=1,
                        dataset=L(CausalActionMixture)(
                            # DROID is the first registered source, not the parent training recipe.
                            datasets=[
                                L(get_causal_action_droid_sft_dataset)(
                                    root="${oc.env:DROID_ROOT}",
                                    fps=15.0,
                                    chunk_length=32,
                                    video_stride="${data_setting.action.video_stride}",
                                    action_space="causal_eef",
                                    statistics_path="${oc.env:ACTION_STATISTICS_PATH}",
                                    block_size_min="${model.config.teacher_forcing_block_size_min}",
                                    block_size_max="${model.config.teacher_forcing_block_size_max}",
                                    history_blocks_min="${model.config.teacher_forcing_history_blocks_min}",
                                    history_blocks_max="${model.config.teacher_forcing_history_blocks_max}",
                                    mode="joint",
                                    use_state=True,
                                    iterable_shuffle=False,
                                    episode_shuffle_seed=42,
                                    use_image_augmentation=False,
                                    use_filter_dict=False,
                                    filter_dict_path=None,
                                    action_normalization=None,
                                    viewpoint="concat_view",
                                    resolution="480",
                                    max_action_dim="${model.config.max_action_dim}",
                                    cfg_dropout_rate=0.1,
                                    tokenizer_config="${model.config.vlm_config.tokenizer}",
                                    format_prompt_as_json=True,
                                    append_viewpoint_info=True,
                                    append_duration_fps_timestamps=True,
                                    append_resolution_info=True,
                                    append_idle_frames=False,
                                    use_success_only=True,
                                    dataset_version=None,
                                    debug_fixed_index=None,
                                    joint_mode_weights=dict(
                                        forward_dynamics=1.0,
                                        inverse_dynamics=1.0,
                                        policy=1.0,
                                    ),
                                )
                            ],
                            weights=[1.0],
                            seed=42,
                        ),
                    ),
                ),
            ),
        ),
        dataloader_val=None,
        upload_reproducible_setup=False,
    ),
    flags={"allow_objects": True},
)

ConfigStore.instance().store(group="experiment", package="_global_", name="action_causal_midtrain_edge", node=cfg)
