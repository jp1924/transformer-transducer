import importlib

from .configuration_transformer_transducer import TransformerTransducerConfig, TransfoXLConfig
from .feature_extraction_transformer_transducer import TransformerTransducerFeatureExtractor
from .modeling_transformer_transducer import TransformerTransducerForRNNT, TransfoXLModel
from .processing_transformer_transduceer import TransformerTransducerProcessor


module = importlib.import_module("transformers")
setattr(module, "TransformerTransducerConfig", TransformerTransducerConfig)
setattr(module, "TransformerTransducerFeatureExtractor", TransformerTransducerFeatureExtractor)
setattr(module, "TransformerTransducerForRNNT", TransformerTransducerForRNNT)
setattr(module, "TransformerTransducerProcessor", TransformerTransducerProcessor)
setattr(module, "TransfoXLConfig", TransfoXLConfig)
setattr(module, "TransfoXLModel", TransfoXLModel)
