"""
    This ComfyUI node provides two model patch nodes that change the built-in 
    CFG guidance function by running a time step prediction like the original
    CFG guidance but on a subset of the whole model, as described in arxiv 
    2508.12880. This subnetwork provides a slightly worse result than the 
    original prediction and by subtracting this worse result it should lead
    to a better guidance to the prompt target.

    @author: orpheus-gaze on Github
"""

import random
import math

from typing_extensions import override

import comfy.model_patcher
import comfy.samplers
from comfy_api.latest import ComfyExtension, io

import torch
import numpy as np


def get_model_config(model):
    """
        Find the correct amount of blocks in a model. If the model is a single 
        stream diffusion model it will only count the double_blocks.
        
        The naming scheme as I have found it to be able to get the amount of 
        layers is as such:
        
        audio_dit layers
        aura  double_layers single_layers
        chroma double_blocks single_blocks
        cogvideo blocks
        anima + cosmos blocks
        flux double_blocks single_blocks
        genmo blocks
        hidream_o1 layers
        hunyuan3d double_blocks single_blocks
        hunyuan3dv2_1 blocks "block"
        hunyuan_video double_blocks single_blocks
        hydit blocks
        kandinsky5 visual_transformer_blocks
        lightricks transformer_blocks
        lumina layers 
        pixartms blocks
        qwen_image transformer_blocks
        wan blocks
        
        Note that I think that hunyuan3dv2_1 is not able to use this function 
        because it replaces its DIT layers with the "block" paradigm instead of
        the usual "double_block" one.
    """
    dif_mod = model.diffusion_model

    double_count = 0
    single_count = 0

    if hasattr(dif_mod, 'double_blocks'):
        # many dit models
        double_count = len(dif_mod.double_blocks)
    if hasattr(dif_mod, 'single_blocks'):
        # many dit models
        single_count = len(dif_mod.single_blocks)

    if hasattr(dif_mod, 'double_layers'):
        # Auraflow DIT
        double_count = len(dif_mod.double_layers)
    if hasattr(dif_mod, 'single_layers'):
        # Auraflow DIT
        single_count = len(dif_mod.single_layers)

    if hasattr(dif_mod, 'layers'):
        # ernie, lumina, omnigen
        double_count = len(dif_mod.layers)

    if hasattr(dif_mod, 'blocks'):
        # cogvideoX, cosmos, anima, genmo, hunyuand3dv2_1, hydit, pixart, wan
        double_count = len(dif_mod.blocks)

    if hasattr(dif_mod, 'visual_transformer_blocks'):
        # kandinsky5
        double_count = len(dif_mod.visual_transformer_blocks)

    if hasattr(dif_mod, 'transformer_blocks'):
        # acestep, lightricks, qwen_image
        double_count = len(dif_mod.transformer_blocks)

    if double_count == 0:
        print("S2 Perpo Guidance was unable to detect layers.")

    return (double_count, single_count)

def skip (args, extra_args):
    """
        The function simply returns args without modification, 
        which means the block's output equals its 
        input—effectively skipping the block's computation entirely.
    """
    return args

def skip_layer_logic(
                        args,
                        double_layers: int,
                        single_layers: int,
                        skip_layers_percentage: int
                    ):
    """
        This methods sets which of the layers should be skipped based on 
        random selection from parameter input.
    """
    model = args["model"]
    cond = args["cond"]
    sigma = args["sigma"]
    model_options = args["model_options"].copy()
    x = args["input"]

    total_num_blocks = double_layers + single_layers

    # skip_layers_percentage is set as user input
    skip_layers = int(math.ceil(skip_layers_percentage / 100.0) * total_num_blocks)
    all_indices = list(range(total_num_blocks))
    layers_to_skip = random.sample(all_indices, skip_layers)

    for layer in layers_to_skip:
        if layer <= double_layers:
            # Do guidance on double blocks
            model_options = comfy.model_patcher.set_model_options_patch_replace(
                                                            model_options,
                                                            skip,
                                                            "dit", 
                                                            "double_block",
                                                            layer
                                                        )
        else:
            # Do guidance on single blocks
            layer = layer - double_layers
            model_options = comfy.model_patcher.set_model_options_patch_replace(
                                                model_options,
                                                skip,
                                                "dit", 
                                                "single_block",
                                                layer
                                            )

    return comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)

class S2GuidanceDIT(io.ComfyNode):
    """
        This class provides S2 Guidance for many DIT models in ComfyUI
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        """
            Defining node parameters.
        """
        return io.Schema(
            node_id="S2Guidance_DIT",
            display_name="✨S2GuidanceDIT",
            category="✨ S2GUIDANCE",
            description=(
                "Enables S²-guidance for certain diffusion models which should \
                lead to better prompt adherence. It achieves this by \
                subtracting a subnetwork of layers from the guidance which \
                should make it more robust than normal CFG-guidance."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("s2_guidance_scale",
                    default=0.25,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "The strength of the S² guidance scale.\
                        \n\n(no effect=0.0, strong effect=2.0, default=0.25)"
                    )
                ),
                io.Int.Input("skip_layers_percentage",
                    default=1,
                    min=1,
                    max=100,
                    step=1,
                    optional=True,
                    tooltip=(
                        "The skip_layers_percentage variable dictates the \
                        percentage of how many layers out of the total that \
                        should be skipped. \n\n(one=1, all=100, default=1)"
                    )
                ),
            ],
            outputs=[
                io.Model.Output("model"),
            ]
        )

    @classmethod
    def execute(
                    cls,
                    model,
                    s2_guidance_scale: float,
                    skip_layers_percentage: int
                ) -> io.NodeOutput:
        '''
            This is an implementation of S²-guidance, as described in arxiv 
            2508.12880, and returns a patched model that applies the guidance 
            during generation by subtracting the weighted (s2_guidance_scale) 
            results of a subnetwork of the model, that is, the calculated 
            predicted condition while layers are removed, from the CFG result.
            
            input: 
                - unpatched model
                - s2_guidance_scale
                - skip_layers_percentage
                
            returns: 
                - S²-guidance patched model
        '''

        def apply_s2_guidance(
                                args,
                                double_layers: int,
                                single_layers: int
                            ):
            (s2_cond_pred,) = skip_layer_logic(
                                            args,
                                            double_layers,
                                            single_layers,
                                            skip_layers_percentage)
            cfg_result = args["denoised"]

            refined = cfg_result - (s2_guidance_scale * s2_cond_pred)
            # Match the 'energy' of the original CFG so colors stay correct
            return refined * (torch.norm(cfg_result, dim=None) / torch.norm(refined, dim=None))

        def post_cfg_function(args):
            model = args["model"]
            (double_layers, single_layers) = get_model_config(model)

            if double_layers == 0:
                print("Model not supported for S²-Guidance")
                # if not supported architecture do not change function
                return io.NodeOutput(m)

            return apply_s2_guidance(args, double_layers, single_layers)

        print(
            "Using S²-Guidance - s2_guidance_scale:", s2_guidance_scale, 
            ", skip_layers_percentage:", skip_layers_percentage)
        m = model.clone()
        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return io.NodeOutput(m)

class PerpoGuidanceDIT(io.ComfyNode):
    """
        This class provides Perpo Guidance for many DIT models in ComfyUI
    """
    @classmethod
    def define_schema(cls) -> io.Schema:
        """
            Defining node parameters.
        """
        return io.Schema(
            node_id="Perpo-Guidance_DIT",
            display_name="✨Perpo-GuidanceDIT",
            category="✨ S2GUIDANCE",
            description=(
                "Enables Perpo-guidance for certain diffusion models which \
                should lead to better prompt adherence. It achieves this by \
                subtracting a subnetwork of layers from the guidance which \
                should make it more robust than normal CFG-guidance."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("perpo_guidance_scale",
                    default=0.3,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "The strength of the perpo guidance scale. \
                        \n\n(no effect=0.0, strong effect=2.0, default=0.3)"
                    )
                ),
                io.Int.Input("skip_layers_percentage",
                    default=1,
                    min=1,
                    max=100,
                    step=1,
                    optional=True,
                    tooltip=(
                        "The skip_layers_percentage variable dictates the \
                        percentage of how many layers out of the total that \
                        should be skipped. \n\n(one=1, all=100, default=1)"
                    )
                ),
                io.Boolean.Input("apply_clamping",
                    default=True,
                    tooltip=(
                        "Set True when you want to clamp the final result.\
                        This will constrain the final result somewhat. \
                        \n\n(no clamping=False, clamping=True, default=True)"
                    )
                ),
            ],
            outputs=[
                io.Model.Output("model"),
            ]
        )

    @classmethod
    def execute(
                cls,
                model,
                perpo_guidance_scale: float,
                skip_layers_percentage: int,
                apply_clamping: bool
                ) -> io.NodeOutput:
        '''
            This is an modified implementation of S²-guidance, as described in 
            arxiv 2508.12880, which I've named as Perpo-Guidance, and returns a 
            patched model that applies the guidance during generation by 
            subtracting the weighted (perpo_guidance_scale) results of a 
            subnetwork of the model, that is, the calculated predicted condition 
            while layers are removed, from the CFG result. The difference from 
            the original implementation (as far as I can tell) is that the 
            original normalises the result for the subtracted network, while 
            here the perpendicular result, that is, the difference of the 
            subnetwork prediction is subtracted from the original cfg result.
            
            input: 
                - unpatched model
                - perpo_guidance_scale
                - skip_layers_percentage
                
            returns: 
                - Perpo-guidance patched model
        '''

        def apply_perpo_guidance(
                                args,
                                double_layers: int,
                                single_layers: int,
                                apply_clamping: bool
                                ):
            '''
                Subtract the perpo-guidance from the CFG result. To make sure 
                that you only subtract the differences from the cfg result you 
                remove the parallel component from the prediction thereby giving 
                you only the orthogonal difference between the cfg prediction 
                and the subnetwork prediction. 
            '''

            cfg_result = args["denoised"]
            (perpo_cond_pred,) = skip_layer_logic(
                                                    args,
                                                    double_layers,
                                                    single_layers,
                                                    skip_layers_percentage)

            cfg_flat = cfg_result.flatten()
            perpo_flat = perpo_cond_pred.flatten()

            # Subtracts the parallel part, leaving only the component that is
            # perpendicular to cfg_result. It isolates the pure difference
            # that isn't already captured by CFG.
            proj = torch.dot(perpo_flat, cfg_flat) / (torch.dot(cfg_flat, cfg_flat) + 1e-6)
            parallel = proj * cfg_result
            orthogonal = perpo_cond_pred - parallel

            refined = cfg_result - (perpo_guidance_scale * orthogonal)

            # Match the 'energy' of the original CFG so colors stay correct
            perpo_result = refined * (torch.norm(cfg_result, dim=None)
                                        / torch.norm(refined, dim=None))

            if apply_clamping is True:
                # Optionally clamp the result to more than 3 sd (~0.99)
                s = torch.quantile(torch.abs(perpo_result), 0.995, dim=None)
                s = torch.clamp(s, min=3.0)
                perpo_result = perpo_result / s * 3.0

            return perpo_result

        def post_cfg_function(args):
            model = args["model"]
            (double_layers, single_layers) = get_model_config(model)

            if double_layers == 0:
                print("Model not supported for Perpo-Guidance")
                return io.NodeOutput(m) # if not supported architecture do not change function

            return apply_perpo_guidance(args, double_layers, single_layers, apply_clamping)

        print("Using Perpo-Guidance - perpo_guidance_scale:",
                perpo_guidance_scale, ", skip_layers_percentage:",
                skip_layers_percentage)

        m = model.clone()
        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return io.NodeOutput(m)

class S2GuidanceDITExtension(ComfyExtension):
    """
        Define the classes in the file
    """
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        """
            Specify the specific nodes available
        """
        return [
            S2GuidanceDIT, PerpoGuidanceDIT
        ]


async def comfy_entrypoint() -> S2GuidanceDITExtension:
    """
        Provide the entrypoint for the extension
    """
    return S2GuidanceDITExtension()
