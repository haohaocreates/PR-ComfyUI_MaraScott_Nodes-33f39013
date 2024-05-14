#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
###
#
#
#
###

from . import __SESSIONS_DIR__, __PROFILES_DIR__
from .py.nodes.AnyBus.AnyBus_v2 import AnyBus_v2 as BusNode_v1
from .py.nodes.AnyBus.AnyBus_v2 import AnyBus_v2 as AnyBusNode_V1
from .py.nodes.AnyBus.AnyBus_v2 import AnyBus_v2 as AnyBusNode_V2
from .py.nodes.AnyBus.AnyBus_v3 import AnyBus_v3 as AnyBusNode_V3
from .py.nodes.AnyBus.ToBasicPipe_v1 import ToBasicPipe_v1 as AnyBusToBasicPipe_v1
from .py.nodes.AnyBus.ToDetailerPipe_v1 import ToDetailerPipe_v1 as AnyBusToDetailerPipe_v1
from .py.nodes.DisplayInfoNode import DisplayInfoNode as DisplayInfoNode_v1
from .py.nodes.UpscalerRefiner.McBoaty_v1 import UpscalerRefiner_McBoaty_v1
from .py.nodes.UpscalerRefiner.McBoaty_v2 import UpscalerRefiner_McBoaty_v2

WEB_DIRECTORY = "./web/assets/js"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "MaraScottAnyBusNode_v3": AnyBusNode_V3,
    "MaraScottAnyBusToBasicPipe_v1": AnyBusToBasicPipe_v1,
    "MaraScottAnyBusToDetailerPipe_v1": AnyBusToDetailerPipe_v1,
    "MaraScottDisplayInfoNode": DisplayInfoNode_v1,
    "MaraScottUpscalerRefinerNode_v2": UpscalerRefiner_McBoaty_v2,

    "MaraScottAnyBusNode": AnyBusNode_V2,

    "MarasitBusNode": BusNode_v1,
    "MarasitUniversalBusNode": BusNode_v1,
    "MarasitAnyBusNode": AnyBusNode_V1,
    "MarasitDisplayInfoNode": DisplayInfoNode_v1,
    "MarasitUpscalerRefinerNode": UpscalerRefiner_McBoaty_v1,
    "MarasitUpscalerRefinerNode_v2": UpscalerRefiner_McBoaty_v2,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes 
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaraScottAnyBusNode_v3": "\ud83d\udc30 AnyBus - UniversalBus v3 /*",
    "MaraScottAnyBusToBasicPipe_v1": "\ud83d\udc30 AnyBus To Basic Pipe v1 /p",
    "MaraScottAnyBusToDetailerPipe_v1": "\ud83d\udc30 AnyBus To Detailer Pipe v1 /p",
    "MaraScottDisplayInfoNode": "\ud83d\udc30 Display Info - Text v1 /i",
    "MaraScottUpscalerRefinerNode_v2": "\ud83d\udc30 Large Refiner - McBoaty v2 /u",

    "MaraScottAnyBusNode": "\u274C AnyBus - UniversalBus v2 /*",

    "MarasitBusNode": "\u274C Bus v1 (deprecated)",
    "MarasitUniversalBusNode": "\u274C Bus - UniversalBus v1 (deprecated)",
    "MarasitAnyBusNode": "\u274C AnyBus - UniversalBus v1 (deprecated)",
    "MarasitDisplayInfoNode": "\u274C Display Info - Text v1 (deprecated)",
    "MarasitUpscalerRefinerNode": "\u274C Large Refiner - McBoaty v1 (deprecated)",
    "MarasitUpscalerRefinerNode_v2": "\u274C Large Refiner - McBoaty v2 (deprecated)",
}

print('\033[34m[Maras IT] \033[92mLoaded\033[0m')
